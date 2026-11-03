// tests/glm_slot_test.cu - a conversation slot's sections against a live split that differs from the one that saved
// it (Glm5SlotLayout, include/strata/core/glm_model.hpp), with made-up records and no model or GPU (the header needs
// the toolkit's headers only):
//  - a slot written under five parts [0,8) [8,12) [12,20) [20,26) [26,32) + the NextN block loads under four parts
//    [0,8) [8,20) [20,26) [26,32): every layer's KDA record, closed-pool records, latents and open pool equal the
//    saved ones; saved again under four parts and loaded under five: equal again, and the bytes of the five-part
//    save are the first slot's;
//  - the head and the section headers are the writer's (Glm5SlotLayout::head / section_header), so the bytes the
//    test writes are the engine's layout;
//  - refused with the cause named: a slot missing a layer's section (the layer named), a slot without the NextN
//    block's records when the live split carries it, a wrong magic, a buffer cut inside the head, inside a section
//    header and inside the payload (truncated, and where), a section with another model's record sizes or layer
//    counts (another model).
#include "strata/core/glm_model.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using strata::core::Glm5SlotLayout;
using Section = Glm5SlotLayout::Section;
using Part = Glm5SlotLayout::Part;
using Piece = Glm5SlotLayout::Piece;
using Sizes = Glm5SlotLayout::Sizes;

namespace {

int g_fail = 0;

#define CHECK(c, ...)                                                                                            \
    do {                                                                                                         \
        if (!(c)) {                                                                                              \
            std::printf("FAIL %s:%d: %s: ", __FILE__, __LINE__, #c);                                             \
            std::printf(__VA_ARGS__);                                                                            \
            std::printf("\n");                                                                                   \
            ++g_fail;                                                                                            \
        }                                                                                                        \
    } while (0)

// the made-up model: 32 layers, every fourth a DSA layer (3, 7, ..., 31), the NextN block as layer 32
constexpr int kLayers = 32;
constexpr int kMtp = kLayers;
constexpr int64_t kPos = 5;
const Sizes kSizes = {6, 4, kPos * 2, 3};   // run, pool_rec, lat, pool (floats)

std::vector<bool> recr_of() {
    std::vector<bool> r((size_t) kLayers);
    for (int il = 0; il < kLayers; ++il) r[(size_t) il] = il % 4 != 3;
    return r;
}

// a layer's records: the KDA state (recurrent layers), or the closed pools, the latents and the open pool (DSA)
struct Layer {
    std::vector<float> kda, pool_rec, lat, pool;
};

std::vector<Layer> model_state(float tag) {
    const std::vector<bool> recr = recr_of();
    std::vector<Layer> st((size_t) kLayers + 1);
    const auto fill = [&](std::vector<float>& v, int64_t n, int il, int kind) {
        v.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i) v[(size_t) i] = tag + (float) (il * 1000 + kind * 100 + i);
    };
    for (int il = 0; il <= kLayers; ++il) {
        if (il < kLayers && recr[(size_t) il]) {
            fill(st[(size_t) il].kda, kSizes.run, il, 1);
        } else {
            fill(st[(size_t) il].pool_rec, kSizes.pool_rec, il, 2);
            fill(st[(size_t) il].lat, kSizes.lat, il, 3);
            fill(st[(size_t) il].pool, kSizes.pool, il, 4);
        }
    }
    return st;
}

std::vector<int> dsa_of(const Part& p) {
    const std::vector<bool> recr = recr_of();
    std::vector<int> r;
    if (p.mtp) r.push_back(kMtp);
    for (int il = p.l0; il < p.l1; ++il)
        if (!recr[(size_t) il]) r.push_back(il);
    return r;
}

// the records of one part's payload in file order, as slot_save writes them and slot_load fills them
std::vector<std::vector<float>*> records_of(const Part& p, std::vector<Layer>& st) {
    const std::vector<bool> recr = recr_of();
    std::vector<std::vector<float>*> r;
    for (int il = p.l0; il < p.l1; ++il)
        if (recr[(size_t) il]) r.push_back(&st[(size_t) il].kda);
    for (int il : dsa_of(p)) r.push_back(&st[(size_t) il].pool_rec);
    for (int il : dsa_of(p)) {
        r.push_back(&st[(size_t) il].lat);
        r.push_back(&st[(size_t) il].pool);
    }
    return r;
}

// slot_save's bytes for a split: the head, then per part its header and payload
std::vector<uint8_t> write_slot(const std::vector<Part>& parts, std::vector<Layer>& st, const Sizes& sz = kSizes) {
    std::vector<uint8_t> out = Glm5SlotLayout::head((uint32_t) parts.size(), kPos);
    const std::vector<bool> recr = recr_of();
    for (const Part& p : parts) {
        int32_t n_rec = 0;
        for (int il = p.l0; il < p.l1; ++il) n_rec += recr[(size_t) il];
        std::vector<int64_t> run_len;
        for (size_t i = 0; i < dsa_of(p).size(); ++i) {
            run_len.push_back(sz.lat);
            run_len.push_back(sz.pool);
        }
        const std::vector<uint8_t> h = Glm5SlotLayout::section_header(p.l0, p.l1, n_rec, sz.run, run_len);
        out.insert(out.end(), h.begin(), h.end());
        for (const std::vector<float>* rec : records_of(p, st)) {
            const size_t at = out.size();
            out.resize(at + rec->size() * sizeof(float));
            std::memcpy(out.data() + at, rec->data(), rec->size() * sizeof(float));
        }
    }
    return out;
}

// slot_load's reading side: the sections parsed, the live parts' records mapped, each record copied in
bool load_slot(const std::vector<uint8_t>& data, const std::vector<Part>& parts, std::vector<Layer>& st,
               std::string& err, std::vector<Section>* secs_out = nullptr) {
    const auto read = [&](uint64_t at, void* p, size_t bytes) {
        if (at + bytes > data.size()) return false;
        std::memcpy(p, data.data() + at, bytes);
        return true;
    };
    std::vector<Section> secs;
    int64_t n_pos = 0;
    if (!Glm5SlotLayout::parse(read, data.size(), kSizes.pool_rec, secs, n_pos, err)) return false;
    if (n_pos != kPos) {
        err = "positions " + std::to_string(n_pos);
        return false;
    }
    std::vector<std::vector<Piece>> pieces;
    if (!Glm5SlotLayout::map(secs, parts, recr_of(), kSizes, pieces, err)) return false;
    if (pieces.size() != parts.size()) {
        err = "pieces for " + std::to_string(pieces.size()) + " parts";
        return false;
    }
    for (size_t i = 0; i < parts.size(); ++i) {
        const std::vector<std::vector<float>*> dst = records_of(parts[i], st);
        if (dst.size() != pieces[i].size()) {
            err = "part " + std::to_string(i) + ": " + std::to_string(pieces[i].size()) + " pieces for " +
                  std::to_string(dst.size()) + " records";
            return false;
        }
        for (size_t k = 0; k < dst.size(); ++k) {
            const Piece& pc = pieces[i][k];
            if (pc.floats != (int64_t) dst[k]->size()) {
                err = "part " + std::to_string(i) + " record " + std::to_string(k) + ": " +
                      std::to_string(pc.floats) + " floats for " + std::to_string(dst[k]->size());
                return false;
            }
            if (!read(pc.at, dst[k]->data(), (size_t) pc.floats * sizeof(float))) {
                err = "record past the end";
                return false;
            }
        }
    }
    if (secs_out != nullptr) *secs_out = secs;
    return true;
}

bool same(const std::vector<Layer>& a, const std::vector<Layer>& b, std::string& where) {
    for (size_t il = 0; il < a.size(); ++il) {
        const Layer& x = a[il];
        const Layer& y = b[il];
        if (x.kda != y.kda || x.pool_rec != y.pool_rec || x.lat != y.lat || x.pool != y.pool) {
            where = "layer " + std::to_string(il);
            return false;
        }
    }
    return true;
}

const std::vector<Part> kFive = {{0, 8, false}, {8, 12, false}, {12, 20, false}, {20, 26, false}, {26, 32, true}};
const std::vector<Part> kFour = {{0, 8, false}, {8, 20, false}, {20, 26, false}, {26, 32, true}};

void test_round_trip() {
    std::vector<Layer> saved = model_state(0.5f);
    const std::vector<uint8_t> five = write_slot(kFive, saved);
    // five parts -> four: every layer's records as saved
    std::vector<Layer> live = model_state(-7.0f);   // (another conversation's state, overwritten by the load)
    std::string err, where;
    std::vector<Section> secs;
    CHECK(load_slot(five, kFour, live, err, &secs), "load 5 -> 4: %s", err.c_str());
    CHECK(same(saved, live, where), "5 -> 4 differs at %s", where.c_str());
    CHECK(secs.size() == 5, "%zu sections", secs.size());
    if (secs.size() == 5) {
        CHECK(secs[1].l0 == 8 && secs[1].l1 == 12 && secs[1].n_rec == 3 && secs[1].run_len.size() == 2,
              "section 1: [%d,%d) n_rec %d runs %zu", secs[1].l0, secs[1].l1, secs[1].n_rec, secs[1].run_len.size());
        CHECK(secs[4].run_len.size() == 6, "section 4: %zu runs (the NextN block's two among them)",
              secs[4].run_len.size());
        CHECK(secs[0].payload == Glm5SlotLayout::kHeadBytes + 16 + 8 + 8 * 4, "section 0 payload at %llu",
              (unsigned long long) secs[0].payload);
        CHECK(secs[4].payload + secs[4].bytes == five.size(), "the last section ends at %llu of %zu",
              (unsigned long long) (secs[4].payload + secs[4].bytes), five.size());
    }
    // the same split: as before
    live = model_state(-7.0f);
    CHECK(load_slot(five, kFive, live, err), "load 5 -> 5: %s", err.c_str());
    CHECK(same(saved, live, where), "5 -> 5 differs at %s", where.c_str());
    // four -> five, and the five-part bytes come back
    live = model_state(-7.0f);
    CHECK(load_slot(five, kFour, live, err), "load 5 -> 4: %s", err.c_str());
    const std::vector<uint8_t> four = write_slot(kFour, live);
    CHECK(four.size() == five.size() - 24, "four-part slot %zu bytes, five-part %zu (one section header fewer)",
          four.size(), five.size());
    std::vector<Layer> again = model_state(-9.0f);
    CHECK(load_slot(four, kFive, again, err), "load 4 -> 5: %s", err.c_str());
    CHECK(same(saved, again, where), "4 -> 5 differs at %s", where.c_str());
    CHECK(write_slot(kFive, again) == five, "the five-part save after the round trip differs");
    // a two-part split and a single part take the same slot
    std::vector<Layer> two = model_state(-1.0f);
    CHECK(load_slot(five, {{0, 20, false}, {20, 32, true}}, two, err), "load 5 -> 2: %s", err.c_str());
    CHECK(same(saved, two, where), "5 -> 2 differs at %s", where.c_str());
    std::vector<Layer> one = model_state(-2.0f);
    CHECK(load_slot(five, {{0, 32, true}}, one, err), "load 5 -> 1: %s", err.c_str());
    CHECK(same(saved, one, where), "5 -> 1 differs at %s", where.c_str());
}

void test_refused() {
    std::vector<Layer> saved = model_state(0.5f);
    std::string err;
    // a layer's section missing: the layer named
    {
        std::vector<Part> four_of_five = kFive;
        four_of_five.erase(four_of_five.begin() + 1);   // [8,12) gone
        const std::vector<uint8_t> s = write_slot(four_of_five, saved);
        std::vector<Layer> live = model_state(-7.0f);
        CHECK(!load_slot(s, kFour, live, err), "a slot without layers 8..11 loaded");
        CHECK(err.find("layer 8") != std::string::npos && err.find("not in the slot") != std::string::npos,
              "missing layer: '%s'", err.c_str());
    }
    // the NextN block's records missing
    {
        std::vector<Part> no_mtp = kFive;
        no_mtp[4].mtp = false;
        const std::vector<uint8_t> s = write_slot(no_mtp, saved);
        std::vector<Layer> live = model_state(-7.0f);
        CHECK(!load_slot(s, kFour, live, err), "a slot without the NextN block loaded");
        CHECK(err.find("layer 32") != std::string::npos && err.find("NextN") != std::string::npos &&
                  err.find("not in the slot") != std::string::npos,
              "missing NextN: '%s'", err.c_str());
        // ... and a live split without the block ignores the slot's
        std::vector<Part> live_no_mtp = kFour;
        live_no_mtp[3].mtp = false;
        const std::vector<uint8_t> with = write_slot(kFive, saved);
        live = model_state(-7.0f);
        CHECK(load_slot(with, live_no_mtp, live, err), "a live split without the NextN block: %s", err.c_str());
        for (int il = 0; il < kLayers; ++il) CHECK(live[(size_t) il].kda == saved[(size_t) il].kda, "layer %d", il);
    }
    // the magic
    {
        std::vector<uint8_t> s = write_slot(kFive, saved);
        s[0] ^= 0x55;
        std::vector<Layer> live = model_state(-7.0f);
        CHECK(!load_slot(s, kFour, live, err), "a wrong magic loaded");
        CHECK(err.find("magic") != std::string::npos, "wrong magic: '%s'", err.c_str());
    }
    // truncated: inside the head, inside a section header, inside the payload, one byte short
    {
        const std::vector<uint8_t> s = write_slot(kFive, saved);
        const size_t cuts[4] = {10, (size_t) Glm5SlotLayout::kHeadBytes + 20, s.size() / 2, s.size() - 1};
        for (size_t cut : cuts) {
            const std::vector<uint8_t> t(s.begin(), s.begin() + (std::ptrdiff_t) cut);
            std::vector<Layer> live = model_state(-7.0f);
            CHECK(!load_slot(t, kFour, live, err), "a slot cut at %zu loaded", cut);
            CHECK(err.find("truncated") != std::string::npos, "cut at %zu: '%s'", cut, err.c_str());
            CHECK(err.find(std::to_string(cut)) != std::string::npos, "cut at %zu, the size not named: '%s'", cut,
                  err.c_str());
        }
    }
    // another model: the KDA record's size, a run's size, the recurrent count
    {
        Sizes other = kSizes;
        other.run = kSizes.run + 1;
        std::vector<Layer> st = saved;
        for (Layer& l : st)
            if (!l.kda.empty()) l.kda.push_back(0.0f);
        std::vector<uint8_t> s = write_slot(kFive, st, other);
        std::vector<Layer> live = model_state(-7.0f);
        CHECK(!load_slot(s, kFour, live, err), "another KDA record size loaded");
        CHECK(err.find("another model") != std::string::npos, "KDA size: '%s'", err.c_str());
        other = kSizes;
        other.lat = kSizes.lat + 2;
        st = saved;
        for (Layer& l : st)
            if (!l.lat.empty()) l.lat.insert(l.lat.end(), 2, 0.0f);
        s = write_slot(kFive, st, other);
        CHECK(!load_slot(s, kFour, live, err), "another latent size loaded");
        CHECK(err.find("another model") != std::string::npos, "latent size: '%s'", err.c_str());
        // one section's range patched: to [0,30), one recurrent layer fewer than its count; to [0,31), one DSA
        // layer fewer than its runs; to [0,33), past the model's layers
        const auto patched_l1 = [&](int32_t l1) {
            std::vector<uint8_t> t = write_slot({{0, 32, true}}, saved);
            std::memcpy(t.data() + Glm5SlotLayout::kHeadBytes + 4, &l1, 4);
            return t;
        };
        CHECK(!load_slot(patched_l1(30), kFour, live, err), "another recurrent count loaded");
        CHECK(err.find("another model") != std::string::npos && err.find("recurrent") != std::string::npos,
              "recurrent count: '%s'", err.c_str());
        CHECK(!load_slot(patched_l1(31), kFour, live, err), "another DSA run count loaded");
        CHECK(err.find("another model") != std::string::npos && err.find("runs") != std::string::npos,
              "DSA run count: '%s'", err.c_str());
        CHECK(!load_slot(patched_l1(33), kFour, live, err), "a range past the layers loaded");
        CHECK(err.find("another model") != std::string::npos && err.find("33") != std::string::npos, "range: '%s'",
              err.c_str());
    }
}

}  // namespace

int main() {
    test_round_trip();
    test_refused();
    if (g_fail != 0) {
        std::printf("glm_slot_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("glm_slot_test: ok\n");
    return 0;
}

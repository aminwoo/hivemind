#include "tools/distill_data.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "common/utils.h"

namespace distill {

namespace {

constexpr std::array<char, 4> MAGIC = {'H', 'D', 'S', 'T'};
constexpr uint32_t VERSION = 2;

}  // namespace

uint16_t half_bits(__half value) {
#if defined(HIVEMIND_BACKEND_TENSORRT) || defined(HIVEMIND_ORT_FP16)
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#else
    // The FP32 backend's __half is a float: copying its first two bytes
    // stored mantissa bits, not a half.
    return hm::float_to_half_bits(value);
#endif
}

void pack_planes(const __half* planes, Record& record) {
    for (int plane = 0; plane < NB_INPUT_CHANNELS; ++plane) {
        const __half* square = planes + plane * 64;
        const float first = __half2float(square[0]);
        bool constant = true;
        uint64_t bits = 0;
        for (int index = 0; index < 64; ++index) {
            const float value = __half2float(square[index]);
            constant = constant && value == first;
            if (value == 1.0f) {
                bits |= uint64_t{1} << index;
            } else if (value != 0.0f) {
                bits = UINT64_MAX;  // not binary
            }
        }
        if (constant) {
            record.planeBits[plane] = 0;
            record.planeValues[plane] = half_bits(square[0]);
        } else if (bits != UINT64_MAX) {
            record.planeBits[plane] = bits;
            record.planeValues[plane] = half_bits(__float2half_rn(1.0f));
        } else {
            throw std::runtime_error("input plane " + std::to_string(plane)
                                     + " is neither constant nor binary");
        }
    }
}

std::vector<Stockfish::Move> policy_actions(Board& board, int boardNumber) {
    std::vector<Stockfish::Move> actions = board.legal_moves(boardNumber);
    std::erase_if(actions, [&board, boardNumber](Stockfish::Move move) {
        return !is_policy_move_representable(board, boardNumber, move);
    });
    actions.push_back(Stockfish::MOVE_NONE);
    return actions;
}

void write_chunk(const std::filesystem::path& path, const std::vector<Record>& records) {
    const std::filesystem::path temporaryPath = path.string() + ".tmp";
    std::ofstream stream(temporaryPath, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Unable to create " + temporaryPath.string());
    }
    auto write = [&stream](const void* data, size_t bytes) {
        stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    };
    const uint64_t count = records.size();
    uint64_t entries = 0;
    for (const Record& record : records) {
        entries += record.policy[0].size() + record.policy[1].size();
    }
    const uint32_t planes = NB_INPUT_CHANNELS;
    write(MAGIC.data(), MAGIC.size());
    write(&VERSION, sizeof(VERSION));
    write(&planes, sizeof(planes));
    write(&count, sizeof(count));
    write(&entries, sizeof(entries));
    for (const Record& r : records) write(r.planeBits.data(), sizeof(r.planeBits));
    for (const Record& r : records) write(r.planeValues.data(), sizeof(r.planeValues));
    for (const Record& r : records) write(&r.value, sizeof(float));
    for (const Record& r : records) write(r.wdl.data(), 3 * sizeof(float));
    for (const Record& r : records) write(&r.movesLeft, sizeof(uint16_t));
    for (const Record& r : records) write(&r.ply, sizeof(uint16_t));
    for (const Record& r : records) write(&r.outcome, sizeof(int8_t));
    for (const Record& r : records) {
        for (int board = 0; board < 2; ++board) {
            const auto size = static_cast<uint16_t>(r.policy[board].size());
            write(&size, sizeof(size));
        }
    }
    for (const Record& r : records) {
        for (int board = 0; board < 2; ++board) {
            for (const PolicyEntry& e : r.policy[board]) write(&e.index, sizeof(uint16_t));
        }
    }
    for (const Record& r : records) {
        for (int board = 0; board < 2; ++board) {
            for (const PolicyEntry& e : r.policy[board]) write(&e.probability, sizeof(uint16_t));
        }
    }
    stream.close();
    if (!stream) {
        throw std::runtime_error("Failed to write " + temporaryPath.string());
    }
    std::filesystem::rename(temporaryPath, path);
}

std::vector<Record> read_chunk(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Unable to open " + path.string());
    }
    auto read = [&stream, &path](void* data, size_t bytes) {
        stream.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
        if (!stream) {
            throw std::runtime_error("Truncated chunk " + path.string());
        }
    };
    std::array<char, 4> magic{};
    uint32_t version = 0;
    uint32_t planes = 0;
    uint64_t count = 0;
    uint64_t entries = 0;
    read(magic.data(), magic.size());
    read(&version, sizeof(version));
    read(&planes, sizeof(planes));
    read(&count, sizeof(count));
    read(&entries, sizeof(entries));
    if (magic != MAGIC || version != VERSION || planes != NB_INPUT_CHANNELS) {
        throw std::runtime_error("Not a readable HDST chunk: " + path.string());
    }
    std::vector<Record> records(count);
    for (Record& r : records) read(r.planeBits.data(), sizeof(r.planeBits));
    for (Record& r : records) read(r.planeValues.data(), sizeof(r.planeValues));
    for (Record& r : records) read(&r.value, sizeof(float));
    for (Record& r : records) read(r.wdl.data(), 3 * sizeof(float));
    for (Record& r : records) read(&r.movesLeft, sizeof(uint16_t));
    for (Record& r : records) read(&r.ply, sizeof(uint16_t));
    for (Record& r : records) read(&r.outcome, sizeof(int8_t));
    uint64_t total = 0;
    for (Record& r : records) {
        for (int board = 0; board < 2; ++board) {
            uint16_t size = 0;
            read(&size, sizeof(size));
            r.policy[board].resize(size);
            total += size;
        }
    }
    if (total != entries) {
        throw std::runtime_error("Inconsistent policy entries in " + path.string());
    }
    for (Record& r : records) {
        for (int board = 0; board < 2; ++board) {
            for (PolicyEntry& e : r.policy[board]) read(&e.index, sizeof(uint16_t));
        }
    }
    for (Record& r : records) {
        for (int board = 0; board < 2; ++board) {
            for (PolicyEntry& e : r.policy[board]) read(&e.probability, sizeof(uint16_t));
        }
    }
    if (stream.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("Trailing bytes in " + path.string());
    }
    return records;
}

ChunkWriter::ChunkWriter(std::filesystem::path directory, std::string prefix, size_t chunkPositions,
                         bool resume)
    : directory_(std::move(directory)), prefix_(std::move(prefix)),
      chunkPositions_(std::max<size_t>(1, chunkPositions)) {
    std::filesystem::create_directories(directory_);
    for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
        if (entry.path().filename().string().find(prefix_ + "_") == 0) {
            if (!resume) {
                throw std::runtime_error("Distillation chunk prefix already exists: " + prefix_);
            }
            ++chunkIndex_;
        }
    }
}

void ChunkWriter::append(Record&& record) {
    records_.push_back(std::move(record));
    if (records_.size() >= chunkPositions_) {
        flush();
    }
}

void ChunkWriter::flush() {
    if (records_.empty()) {
        return;
    }
    std::ostringstream name;
    name << prefix_ << '_' << std::setw(5) << std::setfill('0') << chunkIndex_++ << ".dst";
    write_chunk(directory_ / name.str(), records_);
    records_.clear();
}

}  // namespace distill

#include "RTNeural/RTNeural.h"
#include "dr_wav.h"

#include <iostream>

static void process(const drwav &in_wav, const std::vector<float> in_samples,
                    const uint64_t num_frames, const std::string &model_dir,
                    const std::string &model_filename) {
    std::vector<std::unique_ptr<RTNeural::Model<float>>> models;
    for (uint32_t i = 0; i < in_wav.channels; ++i) {
        // TODO(toms): better way to clone a model?
        std::ifstream json("RTNeural-harness-x/res/" + model_dir + "/" + model_filename + ".json",
                           std::ifstream::binary);

        auto model = RTNeural::json_parser::parseJson<float>(json, true);
        model->reset(); // reset state of network before calling `forward`

        models.push_back(std::move(model));
    }

    auto samples = in_samples; // create a copy of samples for modification

    for (uint32_t f = 0; f < num_frames; ++f) {
        for (uint32_t c = 0; c < in_wav.channels; ++c) {
            float *const sample = &samples[f * in_wav.channels + c];
            *sample = models[c]->forward(sample);

            // HACK(toms): just process 1st channel
            // if (c == 0) {
            //     *sample = models[c]->forward(sample);
            // } else {
            //     *sample = 0.0f;
            // }
        }
    }

    drwav out_wav;
    drwav_data_format format = {
        .container = drwav_container_riff,
        .format = DR_WAVE_FORMAT_PCM,
        .channels = 2,
        .sampleRate = 44100,
        .bitsPerSample = 16,
    };
    drwav_init_file_write(
        &out_wav, ("RTNeural-harness-x/out/" + model_dir + "-" + model_filename + ".wav").c_str(),
        &format, NULL);

    std::vector<int16_t> s16;
    s16.resize(num_frames);

    drwav_f32_to_s16(s16.data(), samples.data(), num_frames * in_wav.channels);

    drwav_uint64 frames_written = drwav_write_pcm_frames(&out_wav, num_frames, s16.data());
    std::cout << "[wav] frames_written: " << frames_written << std::endl;

    drwav_uninit(&out_wav);
}

int main(int argc, char *argv[]) {
    const std::string in_wav_filename = "RTNeural-train-x/res/muff-input-guitar.wav";

    drwav in_wav;
    if (!drwav_init_file(&in_wav, in_wav_filename.c_str(), NULL)) {
        std::cerr << "Error opening and reading WAV file: " << in_wav_filename << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<float> samples(in_wav.totalPCMFrameCount * in_wav.channels);
    uint64_t num_frames =
        drwav_read_pcm_frames_f32(&in_wav, in_wav.totalPCMFrameCount, samples.data());

    std::cout << "[wav] channels: " << in_wav.channels << std::endl;
    std::cout << "[wav] sample_rate: " << in_wav.sampleRate << std::endl;
    std::cout << "[wav] num_frames: " << num_frames << std::endl;

    // Truncate the length of the WAV to avoid overflowing while writing in `dr_wav`
    const uint32_t num_seconds = 10;
    num_frames =
        std::min(num_frames, (uint64_t)(num_seconds * in_wav.sampleRate * in_wav.channels));

    const auto model_filenames = std::vector({
        std::make_pair("aidadsp", "tw40_blues_deluxe_deerinkstudios"),
        std::make_pair("aidadsp", "tw40_blues_solo_deerinkstudios"),
        std::make_pair("aidadsp", "tw40_british_lead_deerinkstudios"),
        std::make_pair("aidadsp", "tw40_british_rhythm_deerinkstudios"),
        std::make_pair("aidadsp", "tw40_california_clean_deerinkstudios"),
        std::make_pair("aidadsp", "tw40_california_crunch_deerinkstudios"),

        std::make_pair("MLTerror15", "0.5-0.5-0.5-model-gru-4"),
        std::make_pair("MLTerror15", "0.5-0.5-0.5-model-gru-6"),
        std::make_pair("MLTerror15", "0.5-0.5-0.5-model-lstm-1"),
        std::make_pair("MLTerror15", "0.5-0.85-0.85-model-gru-5"),
        std::make_pair("MLTerror15", "0.85-0.5-0.85-model-gru-6"),
        std::make_pair("MLTerror15", "0.85-0.85-0.85-model-gru-6"),

        std::make_pair("ds", "tanh_3_256_tf"),

        // "lstm_conv_tf_12",
        // "lstm_tanh_12",
        // "lstm_tf_8",
        // "lstm_tf_12",
        // "tanh_3_256_tf",
        // "Ben Direct Mono.wav_lstm_conv_tf_256",
        // "ts9_test1_in_FP32.wav_lstm_conv_tf_256",
    });

    for (auto [model_dir, model_filename] : model_filenames) {
        process(in_wav, samples, num_frames, model_dir, model_filename);
    }

    drwav_uninit(&in_wav);

    return EXIT_SUCCESS;
}

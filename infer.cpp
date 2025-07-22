#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <cstring>

namespace fs = std::filesystem;

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mnist");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "../mnist_model.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // 입력/출력 이름
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    std::string input_name = input_name_ptr.get();
    std::string output_name = output_name_ptr.get();
    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_name.c_str() };

    std::string test_root = "/home/magon/mnist_png/test/";
    int total = 0;
    int correct = 0;

    std::vector<std::pair<std::string, int>> image_list;

    // 0~9 폴더에서 10장씩만 수집
    for (int label = 0; label <= 9; ++label) {
        std::string label_dir = test_root + std::to_string(label);
        int count = 0;
        for (const auto& entry : fs::directory_iterator(label_dir)) {
            if (entry.path().extension() == ".png") {
                image_list.emplace_back(entry.path().string(), label);
                if (++count >= 10) break;
            }
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& [img_path, gt_label] : image_list) {
        cv::Mat image = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "이미지 로드 실패: " << img_path << std::endl;
            continue;
        }

        cv::resize(image, image, cv::Size(28, 28));
        image.convertTo(image, CV_32FC1, 1.0 / 255.0);

        std::vector<float> input_tensor_values(28 * 28);
        std::memcpy(input_tensor_values.data(), image.ptr<float>(), 28 * 28 * sizeof(float));

        std::vector<int64_t> input_dims = {1, 1, 28, 28};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_dims.data(),
            input_dims.size()
        );

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        int pred = 0;
        float max_val = output_data[0];
        for (int i = 1; i < 10; ++i) {
            if (output_data[i] > max_val) {
                max_val = output_data[i];
                pred = i;
            }
        }

        std::cout << "[GT=" << gt_label << "] 예측=" << pred;
        if (pred == gt_label) {
            std::cout << " ✅" << std::endl;
            correct++;
        } else {
            std::cout << " ❌" << std::endl;
        }
        total++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "\n총 " << total << "장 추론 완료" << std::endl;
    std::cout << "정답률: " << (100.0 * correct / total) << "% (" << correct << "/" << total << ")" << std::endl;
    std::cout << "총 소요 시간: " << total_sec << "초" << std::endl;
    std::cout << "이미지당 평균 시간: " << (1000.0 * total_sec / total) << " ms" << std::endl;

    return 0;
}

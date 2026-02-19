#include "mark_simulator.h"
#include <algorithm>
#include <cmath>

MarkSimulator::MarkSimulator() : noiseGen_(42) {}

// 抗锯齿亚像素矩形绘制
void MarkSimulator::drawSubpixelRect(Matrix2D& img, float x0, float y0,
                                      float x1, float y1, float lineWidth,
                                      float intensity, float asymShift) {
    int rows = img.rows();
    int cols = img.cols();
    float halfW = lineWidth / 2.0f;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float px = static_cast<float>(c);
            float py = static_cast<float>(r);

            // 计算到矩形四条边的距离
            // 顶边
            float dTop = std::fabs(py - y0);
            if (px >= x0 - halfW && px <= x1 + halfW && dTop <= halfW) {
                float coverage = std::max(0.0f, 1.0f - dTop / halfW);
                img.at(r, c) = std::max(img.at(r, c), intensity * coverage);
            }
            // 底边 (含 asymmetry)
            float dBot = std::fabs(py - y1 - asymShift);
            if (px >= x0 - halfW && px <= x1 + halfW && dBot <= halfW) {
                float coverage = std::max(0.0f, 1.0f - dBot / halfW);
                img.at(r, c) = std::max(img.at(r, c), intensity * coverage);
            }
            // 左边
            float dLeft = std::fabs(px - x0);
            if (py >= y0 - halfW && py <= y1 + halfW + asymShift && dLeft <= halfW) {
                float coverage = std::max(0.0f, 1.0f - dLeft / halfW);
                img.at(r, c) = std::max(img.at(r, c), intensity * coverage);
            }
            // 右边 (含 asymmetry)
            float dRight = std::fabs(px - x1 - asymShift);
            if (py >= y0 - halfW && py <= y1 + halfW + asymShift && dRight <= halfW) {
                float coverage = std::max(0.0f, 1.0f - dRight / halfW);
                img.at(r, c) = std::max(img.at(r, c), intensity * coverage);
            }
        }
    }
}

void MarkSimulator::drawBoxMark(Matrix2D& img, float cx, float cy,
                                 float width, float lineWidth, float intensity,
                                 float asymmetryShift) {
    float half = width / 2.0f;
    float x0 = cx - half;
    float y0 = cy - half;
    float x1 = cx + half;
    float y1 = cy + half;

    drawSubpixelRect(img, x0, y0, x1, y1, lineWidth, intensity, asymmetryShift);
}

Matrix2D MarkSimulator::applyStressDeformation(const Matrix2D& img,
                                                 float stressX, float stressY) {
    int rows = img.rows();
    int cols = img.cols();
    Matrix2D result(rows, cols, 0.0f);

    float cx = cols / 2.0f;
    float cy = rows / 2.0f;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float dx = (c - cx) / cx;
            float dy = (r - cy) / cy;

            // 二阶多项式变形模型
            float srcX = c + stressX * (dx * dx - 0.5f) * cx * 0.1f;
            float srcY = r + stressY * (dy * dy - 0.5f) * cy * 0.1f;

            if (srcX >= 0 && srcX < cols - 1 && srcY >= 0 && srcY < rows - 1) {
                result.at(r, c) = img.bilinearInterp(srcX, srcY);
            }
        }
    }
    return result;
}

Matrix2D MarkSimulator::generateTemplate(const MarkSimConfig& config) {
    Matrix2D img(config.imageSize, config.imageSize, config.backgroundIntensity);

    float cx = config.imageSize / 2.0f;
    float cy = config.imageSize / 2.0f;

    // 外框 Box
    drawBoxMark(img, cx, cy, config.markWidth, config.markLineWidth,
                config.markIntensity, 0.0f);

    // 内框 Box
    drawBoxMark(img, cx, cy, config.innerMarkWidth, config.innerLineWidth,
                config.markIntensity, 0.0f);

    return img;
}

Matrix2D MarkSimulator::generateSearchImage(const MarkSimConfig& config) {
    Matrix2D img(config.imageSize, config.imageSize, config.backgroundIntensity);

    float cx = config.imageSize / 2.0f + config.trueOffsetX;
    float cy = config.imageSize / 2.0f + config.trueOffsetY;

    // asymmetry: mark 一侧边缘偏移
    float asymShift = config.asymmetry * config.markLineWidth;

    // 外框 Box (不加 asymmetry)
    drawBoxMark(img, cx, cy, config.markWidth, config.markLineWidth,
                config.markIntensity, 0.0f);

    // 内框 Box (加 asymmetry)
    drawBoxMark(img, cx, cy, config.innerMarkWidth, config.innerLineWidth,
                config.markIntensity, asymShift);

    // 应用 wafer stress 变形
    if (std::fabs(config.stressCoeffX) > 1e-6f || std::fabs(config.stressCoeffY) > 1e-6f) {
        img = applyStressDeformation(img, config.stressCoeffX, config.stressCoeffY);
    }

    // 添加噪声
    noiseGen_.setSeed(config.seed);
    img = noiseGen_.addNoiseWithSNR(img, config.snr);

    return img;
}

std::pair<Matrix2D, Matrix2D> MarkSimulator::generatePair(const MarkSimConfig& config) {
    Matrix2D tmpl = generateTemplate(config);
    Matrix2D search = generateSearchImage(config);
    return {std::move(tmpl), std::move(search)};
}

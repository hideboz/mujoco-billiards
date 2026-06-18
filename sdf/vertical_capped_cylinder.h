// Copyright 2025 Hideaki Sakai

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_PLUGIN_SDF_VERTICAL_CAPPED_CYLINDER_H_
#define MUJOCO_PLUGIN_SDF_VERTICAL_CAPPED_CYLINDER_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjvisualize.h>

#include <optional>

#include "sdf.h"

namespace mujoco::plugin::sdf {

struct VerticalCappedCylinderAttribute {
    // 引数の数
    static constexpr int nattribute = 2;

    // halfheight は円筒の高さの半分。(z軸のプラス方向とマイナス方向にそれぞれこの高さあるため)
    // radius は円筒の半径。ただし、円筒の厚みの中心を通る面から中心までの距離。
    // halfthickness は円筒の厚さの半分。
    static constexpr char const* names[nattribute] = {"halfheight", "radius"};

    // それぞれの属性値のデフォルト値
    static constexpr mjtNum defaults[nattribute] = {0.4, 0.2};
};

class VerticalCappedCylinder {
public:
    // Creates a new VerticalCappedCylinder instance (allocated with `new`) or
    // returns null on failure.
    static std::optional<VerticalCappedCylinder> Create(const mjModel* m, mjData* d, int instance);
    VerticalCappedCylinder(VerticalCappedCylinder&&) = default;
    ~VerticalCappedCylinder() = default;

    void Reset();
    void Visualize(const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn, int instance);
    void Compute(const mjModel* m, mjData* d, int instance);
    mjtNum Distance(const mjtNum point[3]) const;
    void Gradient(mjtNum grad[3], const mjtNum point[3]) const;

    static void RegisterPlugin();

    mjtNum attribute[VerticalCappedCylinderAttribute::nattribute];

private:
    VerticalCappedCylinder(const mjModel* m, mjData* d, int instance);

    SdfVisualizer visualizer_;
};

}  // namespace mujoco::plugin::sdf

#endif  // MUJOCO_PLUGIN_SDF_VERTICAL_CAPPED_CYLINDER_H_

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

#ifndef MUJOCO_PLUGIN_SDF_CHOPPED_CYLINDER_H_
#define MUJOCO_PLUGIN_SDF_CHOPPED_CYLINDER_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjvisualize.h>

#include <optional>

#include "sdf.h"

namespace mujoco::plugin::sdf {

struct ChoppedCylinderAttribute {
    static constexpr int nattribute = 4;

    // halfheight は円筒の高さの半分。(z軸のプラス方向とマイナス方向にそれぞれこの高さあるため)
    // radius は円筒の半径。ただし、円筒の厚みの中心を通る面から中心までの距離。
    // halfthickness は円筒の厚さの半分。
    static constexpr char const* names[nattribute] = {
        "halfheight", "heightdiff",
        "radius", "halfthickness"
    };

    static constexpr mjtNum defaults[nattribute] = {0.4, 0.1, 1.0, 0.02};
};

class ChoppedCylinder {
public:
    // Creates a new ChoppedCylinder instance (allocated with `new`) or
    // returns null on failure.
    static std::optional<ChoppedCylinder> Create(const mjModel* m, mjData* d, int instance);
    ChoppedCylinder(ChoppedCylinder&&) = default;
    ~ChoppedCylinder() = default;

    void Reset();
    void Visualize(const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn, int instance);
    void Compute(const mjModel* m, mjData* d, int instance);
    mjtNum Distance(const mjtNum point[3]) const;
    void Gradient(mjtNum grad[3], const mjtNum point[3]) const;

    static void RegisterPlugin();

    mjtNum attribute[ChoppedCylinderAttribute::nattribute];

private:
    ChoppedCylinder(const mjModel* m, mjData* d, int instance);

    SdfVisualizer visualizer_;
};

}  // namespace mujoco::plugin::sdf

#endif  // MUJOCO_PLUGIN_SDF_CHOPPED_CYLINDER_H_

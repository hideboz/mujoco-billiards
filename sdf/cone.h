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

#ifndef MUJOCO_PLUGIN_SDF_CONE_H_
#define MUJOCO_PLUGIN_SDF_CONE_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjvisualize.h>

#include <optional>

#include "sdf.h"

namespace mujoco::plugin::sdf {

struct ConeAttribute {
    // 引数の数
    static constexpr int nattribute = 2;

    // height は円錐の高さ。
    // radius は円錐の底の半径。
    static constexpr char const* names[nattribute] = {"height", "radius"};

    // それぞれの属性値のデフォルト値
    static constexpr mjtNum defaults[nattribute] = {0.4, 0.2};
};

class Cone {
public:
    // Creates a new Cone instance (allocated with `new`) or
    // returns null on failure.
    static std::optional<Cone> Create(const mjModel* m, mjData* d, int instance);
    Cone(Cone&&) = default;
    ~Cone() = default;

    void Reset();
    void Visualize(const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn, int instance);
    void Compute(const mjModel* m, mjData* d, int instance);
    mjtNum Distance(const mjtNum point[3]) const;
    void Gradient(mjtNum grad[3], const mjtNum point[3]) const;

    static void RegisterPlugin();

    mjtNum attribute[ConeAttribute::nattribute];

private:
    Cone(const mjModel* m, mjData* d, int instance);

    SdfVisualizer visualizer_;
};

}  // namespace mujoco::plugin::sdf

#endif  // MUJOCO_PLUGIN_SDF_CONE_H_

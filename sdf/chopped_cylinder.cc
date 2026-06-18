// Copyright 2025 Hideaki Sakai
//
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

#include "chopped_cylinder.h"

#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>

#include <cstdint>
#include <optional>
#include <utility>

#include "sdf.h"

namespace mujoco::plugin::sdf {
namespace {

static mjtNum distance(const mjtNum point[3], const mjtNum attributes[ChoppedCylinderAttribute::nattribute]) {
    mjtNum hh = attributes[0]; // halfheight
    mjtNum hd = attributes[1]; // heightdiff
    mjtNum r = attributes[2]; // radius
    mjtNum ht = attributes[3]; // halfthickness

    mjtNum X = point[0];
    // mjtNum Y = point[1];
    mjtNum Z = point[2];
    // mjtNum absZ = mju_abs(Z);  // z の絶対値
    mjtNum dXY = mju_norm(point, 2);  // sqrt{x^2 + y^2}

    // 円筒側面からの距離 (内部がマイナス)
    mjtNum dSide = mju_abs(dXY - r) - ht;

    // 円筒下側のキャップからのZ方向の距離 (下側は斜めでなく、垂直) (内部がマイナス)
    mjtNum dCap = -Z - hh;

    // 円筒上部の斜めのキャップからの距離 (内部がマイナス)
    mjtNum dChopped = Z - hh - (hd * (X - r -ht) / (2 * (r + ht)));

    // 円筒の厚みの内部はマイナス、それ以外はプラスの値をとる
    mjtNum dist = mju_max(mju_max(dSide, dCap), dChopped);

    return dist;
}

}  // namespace

// factory function
std::optional<ChoppedCylinder> ChoppedCylinder::Create(const mjModel* m, mjData* d, int instance) {
    if (AllAttributesExist<ChoppedCylinderAttribute>(m, instance)) {
        return ChoppedCylinder(m, d, instance);
    } else {
        mju_warning("Invalid parameter specification in ChoppedCylinder plugin");
        return std::nullopt;
    }
}

// plugin constructor
ChoppedCylinder::ChoppedCylinder(const mjModel* m, mjData* d, int instance) {
    SdfDefault<ChoppedCylinderAttribute> defattribute;

    for (int i = 0; i < ChoppedCylinderAttribute::nattribute; i++) {
        attribute[i] = defattribute.GetDefault(
            ChoppedCylinderAttribute::names[i], mj_getPluginConfig(m, instance, ChoppedCylinderAttribute::names[i]));
    }
}

// add new element in the vector storing iteration counts
void ChoppedCylinder::Compute(const mjModel* m, mjData* d, int instance) { visualizer_.Next(); }

// reset visualization counter
void ChoppedCylinder::Reset() { visualizer_.Reset(); }

// plugin visualization
void ChoppedCylinder::Visualize(const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn,
                               int instance) {
    visualizer_.Visualize(m, d, opt, scn, instance);
}

// sdf
mjtNum ChoppedCylinder::Distance(const mjtNum point[3]) const { return distance(point, attribute); }

// gradient of sdf
void ChoppedCylinder::Gradient(mjtNum grad[3], const mjtNum point[3]) const {
    mjtNum eps = 1e-8;
    mjtNum dist0 = distance(point, attribute);

    mjtNum pointX[3] = {point[0] + eps, point[1], point[2]};
    mjtNum distX = distance(pointX, attribute);
    mjtNum pointY[3] = {point[0], point[1] + eps, point[2]};
    mjtNum distY = distance(pointY, attribute);
    mjtNum pointZ[3] = {point[0], point[1], point[2] + eps};
    mjtNum distZ = distance(pointZ, attribute);

    grad[0] = (distX - dist0) / eps;
    grad[1] = (distY - dist0) / eps;
    grad[2] = (distZ - dist0) / eps;
}

// plugin registration
void ChoppedCylinder::RegisterPlugin() {
    mjpPlugin plugin;
    mjp_defaultPlugin(&plugin);

    plugin.name = "mujoco.sdf.chopped_cylinder";
    plugin.capabilityflags |= mjPLUGIN_SDF;

    plugin.nattribute = ChoppedCylinderAttribute::nattribute;
    plugin.attributes = ChoppedCylinderAttribute::names;
    plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

    plugin.init = +[](const mjModel* m, mjData* d, int instance) {
        auto sdf_or_null = ChoppedCylinder::Create(m, d, instance);
        if (!sdf_or_null.has_value()) {
            return -1;
        }
        d->plugin_data[instance] = reinterpret_cast<uintptr_t>(new ChoppedCylinder(std::move(*sdf_or_null)));
        return 0;
    };
    plugin.destroy = +[](mjData* d, int instance) {
        delete reinterpret_cast<ChoppedCylinder*>(d->plugin_data[instance]);
        d->plugin_data[instance] = 0;
    };
    plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data, int instance) {
        auto sdf = reinterpret_cast<ChoppedCylinder*>(plugin_data);
        sdf->Reset();
    };
    plugin.visualize =
        +[](const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn, int instance) {
            auto* sdf = reinterpret_cast<ChoppedCylinder*>(d->plugin_data[instance]);
            sdf->Visualize(m, d, opt, scn, instance);
        };
    plugin.compute = +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        auto* sdf = reinterpret_cast<ChoppedCylinder*>(d->plugin_data[instance]);
        sdf->Compute(m, d, instance);
    };
    plugin.sdf_distance = +[](const mjtNum point[3], const mjData* d, int instance) {
        auto* sdf = reinterpret_cast<ChoppedCylinder*>(d->plugin_data[instance]);
        return sdf->Distance(point);
    };
    plugin.sdf_gradient =
        +[](mjtNum gradient[3], const mjtNum point[3], const mjData* d, int instance) {
            auto* sdf = reinterpret_cast<ChoppedCylinder*>(d->plugin_data[instance]);
            sdf->visualizer_.AddPoint(point);
            sdf->Gradient(gradient, point);
        };
    plugin.sdf_staticdistance = +[](const mjtNum point[3], const mjtNum* attributes) {
        return distance(point, attributes);
    };
    plugin.sdf_aabb = +[](mjtNum aabb[6], const mjtNum* attributes) {
        mjtNum halfheight = attributes[0];
        // mjtNum heightdiff = attributes[1];
        mjtNum radius = attributes[2];
        mjtNum halfthickness = attributes[3];
        aabb[0] = aabb[1] = aabb[2] = 0;
        aabb[3] = aabb[4] = radius + halfthickness;
        aabb[5] = halfheight;
    };
    plugin.sdf_attribute = +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<ChoppedCylinderAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
    };

    mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf

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

#include "hollow_cylinder.h"

#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>

#include <cstdint>
#include <optional>
#include <utility>

#include "sdf.h"

namespace mujoco::plugin::sdf {
namespace {

static mjtNum distance(const mjtNum point[3], const mjtNum attributes[HollowCylinderAttribute::nattribute]) {
    mjtNum halfheight = attributes[0];
    mjtNum radius = attributes[1];
    mjtNum halfthickness = attributes[2];

    mjtNum absZ = mju_abs(point[2]);  // z の絶対値
    mjtNum dXY = mju_norm(point, 2);  // sqrt{x^2 + y^2}

    // 円筒側面からの距離 (内部がマイナス)
    mjtNum dSide = mju_abs(dXY - radius) - halfthickness;

    // 円筒上下のキャップからのZ方向の距離 (内部がマイナス)
    mjtNum dCap = absZ - halfheight;

    // 円筒の厚みの内部はマイナス、それ以外はプラスの値をとる
    mjtNum dist = mju_max(dSide, dCap);

    return dist;
}

}  // namespace

// factory function
std::optional<HollowCylinder> HollowCylinder::Create(const mjModel* m, mjData* d, int instance) {
    if (AllAttributesExist<HollowCylinderAttribute>(m, instance)) {
        return HollowCylinder(m, d, instance);
    } else {
        mju_warning("Invalid parameter specification in HollowCylinder plugin");
        return std::nullopt;
    }
}

// plugin constructor
HollowCylinder::HollowCylinder(const mjModel* m, mjData* d, int instance) {
    SdfDefault<HollowCylinderAttribute> defattribute;

    for (int i = 0; i < HollowCylinderAttribute::nattribute; i++) {
        attribute[i] = defattribute.GetDefault(
            HollowCylinderAttribute::names[i], mj_getPluginConfig(m, instance, HollowCylinderAttribute::names[i]));
    }
}

// add new element in the vector storing iteration counts
void HollowCylinder::Compute(const mjModel* m, mjData* d, int instance) { visualizer_.Next(); }

// reset visualization counter
void HollowCylinder::Reset() { visualizer_.Reset(); }

// plugin visualization
void HollowCylinder::Visualize(const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn,
                               int instance) {
    visualizer_.Visualize(m, d, opt, scn, instance);
}

// sdf
mjtNum HollowCylinder::Distance(const mjtNum point[3]) const { return distance(point, attribute); }

// gradient of sdf
void HollowCylinder::Gradient(mjtNum grad[3], const mjtNum point[3]) const {
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
void HollowCylinder::RegisterPlugin() {
    mjpPlugin plugin;
    mjp_defaultPlugin(&plugin);

    plugin.name = "mujoco.sdf.hollow_cylinder";
    plugin.capabilityflags |= mjPLUGIN_SDF;

    plugin.nattribute = HollowCylinderAttribute::nattribute;
    plugin.attributes = HollowCylinderAttribute::names;
    plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

    plugin.init = +[](const mjModel* m, mjData* d, int instance) {
        auto sdf_or_null = HollowCylinder::Create(m, d, instance);
        if (!sdf_or_null.has_value()) {
            return -1;
        }
        d->plugin_data[instance] = reinterpret_cast<uintptr_t>(new HollowCylinder(std::move(*sdf_or_null)));
        return 0;
    };
    plugin.destroy = +[](mjData* d, int instance) {
        delete reinterpret_cast<HollowCylinder*>(d->plugin_data[instance]);
        d->plugin_data[instance] = 0;
    };
    plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data, int instance) {
        auto sdf = reinterpret_cast<HollowCylinder*>(plugin_data);
        sdf->Reset();
    };
    plugin.visualize =
        +[](const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn, int instance) {
            auto* sdf = reinterpret_cast<HollowCylinder*>(d->plugin_data[instance]);
            sdf->Visualize(m, d, opt, scn, instance);
        };
    plugin.compute = +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        auto* sdf = reinterpret_cast<HollowCylinder*>(d->plugin_data[instance]);
        sdf->Compute(m, d, instance);
    };
    plugin.sdf_distance = +[](const mjtNum point[3], const mjData* d, int instance) {
        auto* sdf = reinterpret_cast<HollowCylinder*>(d->plugin_data[instance]);
        return sdf->Distance(point);
    };
    plugin.sdf_gradient =
        +[](mjtNum gradient[3], const mjtNum point[3], const mjData* d, int instance) {
            auto* sdf = reinterpret_cast<HollowCylinder*>(d->plugin_data[instance]);
            sdf->visualizer_.AddPoint(point);
            sdf->Gradient(gradient, point);
        };
    plugin.sdf_staticdistance = +[](const mjtNum point[3], const mjtNum* attributes) {
        return distance(point, attributes);
    };
    plugin.sdf_aabb = +[](mjtNum aabb[6], const mjtNum* attributes) {
        mjtNum halfheight = attributes[0];
        mjtNum radius = attributes[1];
        mjtNum halfthickness = attributes[2];
        aabb[0] = aabb[1] = aabb[2] = 0;
        aabb[3] = aabb[4] = radius + halfthickness;
        aabb[5] = halfheight;
    };
    plugin.sdf_attribute = +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<HollowCylinderAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
    };

    mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf

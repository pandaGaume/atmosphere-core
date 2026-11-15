# WebGPU sky integration contract

Status: implemented runtime contract for milestone 1, **integrable sky**. The public entry is `@spacexr/atmosphere/webgpu`. Numerical qualification is case-specific and recorded by `npm run test:webgpu`; it is not an observational or performance guarantee. See [the implementation plan](INTEGRATION.md) and [the precision protocol](WEBGPU-PRECISION.md).

## Package boundary

The numerical core stays in `src`, without DOM, WebGPU or Babylon dependencies. The renderer lives under `webgpu/` and be built as the separate `@spacexr/atmosphere/webgpu` entry. Importing the package root must not load that entry. Shader sources live under `webgpu/shaders/`; the build embeds their named modules without runtime fetches or source rewriting. Generated TypeScript modules are build artifacts.

The first renderer supports the existing spectral path: 24 or 48 midpoint wavelengths, one to four analytic layers, single scattering and the existing phase tables. Unsupported profiles, layer counts and device limits fail explicitly. The historical two-profile RGB renderer remains a laboratory comparison, not the public integration path. Arbitrary custom `IAtmosphere` implementations must obey the analytic profile and optics contract; custom density functions that differ from those profiles are not silently approximated.

## Responsibilities

| Concern                 | Atmosphere renderer                                  | Host engine                                                              |
| ----------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------ |
| Device                  | Borrows the supplied device                          | Creates, owns and replaces the device                                    |
| Canvas and presentation | No canvas, context configuration or swapchain access | Configures the context and presents                                      |
| LUT construction        | Encodes compute passes into the supplied encoder     | Calls outside any open pass, orders work and submits                     |
| Sky drawing             | Supplies WGSL readers and a bind group layout        | Creates its pipeline and draws into its own pass                         |
| GPU resources           | Owns its buffers, textures, views and samplers       | Borrows read-only bindings, releases dependent groups before replacement |
| Coordinates             | Computes physical quantities in ECEF conventions     | Converts engine positions and directions to ECEF                         |
| Display                 | Returns scene-linear radiance                        | Applies exposure, tone mapping, gamut mapping and display encoding       |
| Device loss             | Becomes unusable, exposes the failure                | Recreates the renderer on the replacement device                         |

Creation may compile pipelines asynchronously. Normal frame encoding never calls `queue.submit`, `onSubmittedWorkDone`, `mapAsync` or a readback. Uploading immutable tables during creation is allowed. Diagnostics that submit or read back work belong to an explicit test harness, not the frame path.

## Physical snapshot and sunlight

`create(device, atmosphere, options)` captures the ellipsoid axes, top altitude, analytic profiles, resolved spectral optical coefficients, phase tables and unit ECEF direction toward the star. The snapshot no longer reads caller-owned objects during rendering. Invalid or non-finite coefficients are rejected before resource allocation where possible; partial creation failures release allocated resources.

`atmosphere.sun` is the only source of the solar direction. Options select the incident spectrum and wavelength count through the existing `makeSpectralScene` configuration subset: `source`, `starTemperature`, `distanceAU`, `bandCenter`, `bandWidth` and `wavelengths`. Geometry, composition and layer overrides in the broader physical-config type are not accepted here. The contract does not add a second sun override.

For version 1, changing sunlight, physical state, spectrum or LUT quality requires creating a replacement renderer. The host can prepare it while retaining the previous instance, then switch after encoding the replacement LUTs. Mutation of a previously supplied atmosphere never updates the current snapshot. Dynamic shared caches and incremental physical updates are later optimizations.

## Camera input

`IAtmosphereCamera` contains the following fields. All arrays are copied during `update`; vectors are finite three-component tuples.

| Field                             | Meaning                                                                         |
| --------------------------------- | ------------------------------------------------------------------------------- |
| `positionEcefMeters`              | CPU JavaScript-number ECEF position, retained in double precision until packing |
| `forwardEcef`                     | Unit direction from the camera into the scene                                   |
| `rightEcef`                       | Unit screen-right direction                                                     |
| `upEcef`                          | Unit screen-up direction                                                        |
| `halfWidthTangent`                | Positive tangent of half the horizontal field of view                           |
| `halfHeightTangent`               | Positive tangent of half the vertical field of view                             |
| `viewportWidth`, `viewportHeight` | Positive integer physical pixel dimensions                                      |

ECEF is right handed: X at latitude 0 and longitude 0, Y at latitude 0 and longitude 90 degrees east, Z toward the north pole. The camera basis is orthonormal, with `cross(rightEcef, upEcef) = -forwardEcef`. Components must be unit length and pairwise orthogonal within `1e-6`; invalid bases are rejected rather than silently reoriented. Other angular inputs use radians. Public distances use metres.

For normalized viewport coordinates `(u, v)`, with `(0, 0)` at the top left, the unjittered ray is:

```text
normalize(forwardEcef
  + (2*u - 1)*halfWidthTangent*rightEcef
  + (1 - 2*v)*halfHeightTangent*upEcef)
```

The host reconstructs its actual fragment rays, including any asymmetric or jittered projection, and supplies ECEF directions to the sky reader. The camera tangents describe the unjittered symmetric envelope for diagnostics; they do not replace the host's projection matrix. Orthographic cameras are outside version 1.

Cameras on the ellipsoid or above it are supported, including outside the atmosphere. A geodetically underground camera is rejected. A negative height within 10 micrometres caused by surface-conversion roundoff is snapped to one micrometre above the geodetic surface before horizon construction. The atmosphere's geodetic top defines the inside/outside regime, with equality assigned to the outside path. A transition invalidates the view construction explicitly. Neither regime may integrate density outside the physical profile support. The numerical acceptance domain is established by the precision protocol, not by this input schema alone.

## Runtime surface and frame ordering

The runtime surface is:

```ts
const sky = await AtmosphereSky.create(device, atmosphere, options);
sky.update(camera); // CPU snapshot and invalidation only

function frame() {
    const encoder = device.createCommandEncoder();
    const ready = sky.encode(encoder); // one construction task by default
    if (ready) {
        const resources = sky.bindings();
        // Host draws with resources in a later pass on this encoder.
    }
    device.queue.submit([encoder.finish()]);
    // Schedule the next frame while ready is false.
}
// After the final submission referencing the instance:
// sky.dispose();
```

`update` has no GPU side effects. It validates the input, computes geodetic metadata through `@spacexr/geodesy`, snapshots the camera and marks dependent resources dirty. Several updates before encoding collapse to the latest state.

`encode(encoder, maxDispatches = 1)` returns a boolean indicating whether the readers are ready downstream of this encoder. It processes a bounded number of construction tasks without waiting. The dispatch budget must be an integer from 1 to 4096. Increasing it is a host decision based on measured GPU cost. Initial solar construction is progressive; subsequent translations rebuild the view table.

`encode` requires a camera and an encoder from the borrowed device, outside an active pass. It encodes parameter uploads before their dependent computations, immutable-LUT construction on first use, then the required view construction. Frame-specific uploads must preserve encoder ordering, for example through staging copies. Repeated updates must not use `queue.writeBuffer` to overwrite parameters still referenced by an unsubmitted command buffer.

One renderer instance represents one view. For version 1, each `update -> encode -> optional draw -> submit` sequence must be submitted before the next sequence on that instance. The host must not discard an encoded construction and then continue using the instance as though it had been submitted. If an encoder is abandoned, recreate that instance. Submission does not require waiting for GPU completion. Command buffers on the borrowed device's queue execute in the submitted order.

Until `encode` returns true, `bindings()` throws a clear not-ready error, including after translation. When ready, it exposes resources valid for downstream reads in that submitted sequence; encoding does not imply GPU completion. Initialization is ordered by the command stream. A host must not sample the textures before their construction.

With a small dispatch budget, keep the selected camera position stable while its construction finishes. The example queues the latest requested camera and applies it after the current construction. Updating the position during construction restarts the view tasks for the new position. It does not enable approximate translation reuse. Continuous motion at XR frame rates has not been qualified.

### Invalidation

| Change                                  | Version 1 action                                                    |
| --------------------------------------- | ------------------------------------------------------------------- |
| First camera                            | Construct solar resources and Sky-View                              |
| Any camera translation                  | Recompute local geometry, horizon and Sky-View                      |
| Inside/outside transition               | Select the appropriate construction path and rebuild                |
| Rotation or projection                  | Refresh camera metadata; retain the position-dependent full-sky LUT |
| Viewport resize                         | Refresh metadata; retain fixed-quality full-sky allocation          |
| Atmosphere, sunlight, spectrum, quality | Create a replacement instance                                       |
| Device loss                             | Create a replacement instance on a new device                       |

No translation reuse is enabled in version 1. Reuse budgets and calibration are deliberately deferred until the spectral path is validated.

## Reader ABI and resources

`bindings()` returns `IAtmosphereSkyBindings`: a renderer-owned `GPUBindGroupLayout`, a `GPUBindGroup`, a resource `generation` fixed at 1 for the instance, and named read-only texture views. Compare the owner as well as the generation; dimensions and formats are documented below. The host uses that explicit layout in its own pipeline layout. It must not infer layouts from an unrelated automatically laid-out pipeline.

The renderer supplies a complete WGSL reader module for group 0, with a reserved `atmosphere_` namespace. Group 0 is reserved by the host pipeline for this module; other engine resources use subsequent groups. Arbitrary group remapping is outside version 1. Bindings, formats, array dimensions, uniform alignment and the packed spectral stride must be published with the implementation and checked against generated WGSL. The historical 112-byte and layered 304-byte spectral records must never be interchanged.

Version 1 allocates persistent resources at creation; bounded staging uploads are retired after their command buffers have been submitted. It and keeps their layout and generation stable for the instance lifetime. Viewport resizing does not change LUT resolution. The host obtains a fresh binding set when replacing an instance. Borrowed resources must not be destroyed, written to or retained after disposal. No consumer needs internal texture coordinates to draw the sky.

The reader module exposes these names:

```wgsl
fn atmosphere_skyRadiance(directionEcef: vec3f) -> vec3f
fn atmosphere_transmittanceToSun(
    pRelativeToCameraMeters: vec3f, wavelengthGroup: u32
) -> vec3f
```

The direction must be finite and nonzero; the reader normalizes it. `skyRadiance` returns atmospheric path radiance up to the first ground intersection or atmospheric exit. It excludes a synthetic ground color and the stellar disk. A ray missing the atmosphere returns zero. The host draws ground geometry and any stellar disk separately, using the same radiometric normalization if compositing a disk.

`transmittanceToSun` returns three consecutive wavelength transmittances in `[0, 1]`, not an RGB tint. `wavelengthGroup` must be less than the wavelength count divided by three, and positions must be on or above the ellipsoid. Planet-blocked solar rays return zero; an unblocked segment that never enters the atmosphere returns one. The public solar reader performs direct spectral integration. Sky construction uses cached solar columns with a direct-integration fallback when shadowed interpolation corners invalidate the lookup. The longitude coordinate follows the solar tangent of the ellipsoid at each latitude and altitude, rather than the sea-level terminator. All longitude nodes cover illuminated paths, clustered toward that tangent; analytic ground occultation remains authoritative. Sky tiles in one encoding budget share a dispatch, with the spectral group and row tile addressed by the workgroup z coordinate. The budget still counts construction tasks, including partial final row tiles. Relative position components are along ECEF axes; they are not local east/north/up components. The packer retains sufficient CPU origin information for this evaluation, as qualified by the precision protocol.

The bindings describe a solar optical-depth resource, spectral Sky-View scattering resources, view-path transmission resources, phase tables and the associated camera/model metadata. The reader layout is explicit and stable:

| Binding | Resource               | Layout                                                                                        |
| ------- | ---------------------- | --------------------------------------------------------------------------------------------- |
| 0       | Frame parameters       | 160-byte uniform: 8 float vec4 records, then quality and size uint vec4 records               |
| 1       | Layer profiles         | Read-only storage, 32 bytes per layer, kilometre bounds and shape                             |
| 2       | Spectral coefficients  | Read-only storage, 304 bytes per wavelength triple, inverse-kilometre coefficients            |
| 3       | Phase functions        | Unfilterable `rgba32float` 2d texture, `512 x (groups * layers)`                              |
| 4       | Sky radiance           | `rgba16float` 2d-array, `width x (2 * hemisphereRows) x groups`                               |
| 5       | View-path transmission | Same format and dimensions as radiance                                                        |
| 6       | Sky sampler            | Linear filtering, repeat azimuth, clamp elevation                                             |
| 7       | Horizon elevations     | Read-only storage, ground tangent then outer-atmosphere tangent, each one float32 per azimuth |
| 8       | Solar columns          | Unfilterable `rgba32float` 3d texture, `64 x 32 x 32`, four layer columns in kilometres       |

Solar texels with a negative first component represent blocked samples. Other solar texels contain density columns, not RGB transmittance. The azimuthal solar map concentrates samples near the terminator; altitude is quadratic. Sky elevation is quartic below the ellipsoidal horizon and for inside-atmosphere upward rays. Outside the atmosphere, the upper side uses a quadratic mapping between the ground and outer-atmosphere tangents, concentrating samples on the visible limb. Tangent texels evaluate the two limiting path lengths explicitly, and a `2e-7` radian reader band snaps to those limits to match the declared float32 direction uncertainty.

Default quality is width 64, 64 rows per hemisphere, 256 view steps and 192 solar steps. These values match the qualification profile. Lower quality options change the numerical error and require their own validation.

## Color and physical limits

Sky radiance matches `integrateSpectral(...).radiance`: CIE-integrated scene-linear sRGB, normalized to the existing reference solar luminance. It is not sRGB encoded and not an absolute SI radiance RGB triplet. Conversion may produce negative out-of-gamut components; the renderer must not silently clamp, expose or tone-map them before numerical validation. The normalization and source spectrum used by the CPU and GPU must be identical and recorded.

Solar transmittance is spectral. A spectrum-weighted RGB reconstruction is not three independent physical transmittances. `AtmosphereAerialVolume` defines an explicit solar-spectrum-weighted approximation for RGB materials: integrate transmitted incident spectral power into RGB, then divide each channel by its vacuum incident white point. Vacuum transmission is exactly one. Apply this approximation only to scene-linear material RGB, before display encoding. It cannot recover an arbitrary material reflectance spectrum.

Finite-segment integration skips a midpoint only when every local layer density is exactly zero. Extinction and every scattering source are then exactly zero, so the step has unit transmission and zero radiance regardless of incident sunlight. This avoids tracing solar columns and sampling diffuse illumination for empty positions, including vacuum near the conservative ellipsoidal envelope. There is no density threshold, reduced step count or altered wavelength sampling.

The physical model remains single scattering, with the documented molecular and particle assumptions. Numerical agreement is not observational calibration. Optical coefficients must not be tuned to force a target sky color.

## Lifetime, limits and failures

Creation checks required texture/storage formats, dimensions, array layers, binding counts and buffer limits against the supplied device. It neither requests a replacement device nor enables optional features on the host's behalf. No timestamp-query feature is required. Compilation and allocation failures reject creation with diagnostics and resource cleanup.

`dispose()` is idempotent. It releases renderer-owned resources without destroying the device. Call it only after the final command buffers using those resources have been submitted; no CPU wait for their GPU completion is required. Operations other than repeated disposal after disposal fail synchronously. Device loss invalidates the renderer; encoding then fails when loss has been observed, and the host also monitors `device.lost`. Previously encoded work has no completion guarantee after loss.

Each XR eye uses its own instance in version 1, including independent metadata buffers, horizon data and Sky-View textures. They may borrow the same device. Solar-resource sharing is deferred until ownership and reference counting are implemented. A test must encode and draw both eyes before a single host submission to detect accidental cross-view overwrites.

## Acceptance and implementation sequence

1. Implement the separate entry, typed camera/resource interfaces and shader build without changing root-import requirements.
2. Extract spectral solar and Sky-View construction, with explicit layouts and host-owned submission.
3. Implement translation and compensated geometry using the precision protocol; keep reuse disabled.
4. Make the laboratory consume the public entry, retaining its UI, display pipeline and diagnostics.
5. Install an `npm pack` tarball into an independent minimal consumer. Render translated and rotated cameras, test outside/inside crossings and two views, and run the independent CPU/GPU suite in headless Chrome CI.

These steps are implemented by the separate entry, laboratory adapter, packed consumer and browser harness. Run both `npm run check` and `npm run test:webgpu` when changing the renderer. The browser job fails if WebGPU is missing; the repository CI workflow runs the same command and saves its report and screenshot. Finite-distance camera volumes are also implemented and checked against the independent CPU reference in the browser harness.

## Finite-distance aerial transport

After encoding a sky snapshot, call `await sky.createAerialVolume(options)` once. The volume borrows that sky instance and its immutable atmospheric model, solar data and current observer position. Call `volume.encode(encoder, camera)` after encoding the matching sky position and before host submission. `bindings()` exposes filterable `rgba16float` 3D radiance and transmission views plus dimensions. Creation and encoding never submit command buffers. Dispose the volume before its parent sky. Rotation invalidates the view volume; an unchanged camera reuses it.

The x/y endpoints represent full-viewport perspective rays. Each ray's depth domain is its visible atmospheric interval, ending at the first reference-ground intersection or atmospheric exit. Vacuum before entry and opaque ground behind that intersection do not consume integration samples. For an external observer the normalized depth parameter `z` maps to a physical interval fraction `z * (2 - z)`, concentrating samples near the dense surface end. For an internal observer it maps to `z * z`, concentrating samples near the eye. The existing segment integrator and physical coefficients are unchanged.

Include `atmosphereAerialDepthWgsl` and use `atmosphere_aerialSampleDepth(origin, rayPerKm, envelopeScale, distanceKm, depthCount)` to obtain the depth texture coordinate, already mapped to texel centers. It interpolates cumulative transport in physical distance between stored slices. The scaled origin and ray must use the same ellipsoid axes as construction. A bare square-root mapping against the complete outer shell is no longer valid.

Angular filtering must respect the endpoint geometry. When rendering a planetary surface, reproject its endpoint onto the same scaled ellipsoidal level along each neighbouring volume ray, then evaluate that ray's depth coordinate. Rays missing the endpoint level are excluded and the valid angular weights are normalized. Using one camera distance for every corner mixes vacuum with dense near-ground air and creates artificial rings. Finite planes use their own intersection geometry instead; their neighbouring distances must not be treated as planetary levels. Compose scene-linear `surface * transmission + radiance`.

The volume accumulates the same spectral transport equations over successive finite segments. Wavelength triples run concurrently in the same workgroup, each using the common finite-segment integrator. A bounded workgroup reduction combines their XYZ radiance and transmission into the existing RGB volume. Dimensions and segment steps still bound memory and work; defaults remain 32 by 18 by 16, with two steps per segment. No additional pass or resource is introduced by the corrected parameterization.

`integrateSpectral` accepts optional `maxDistanceMeters` for an independent finite-path CPU reference. Zero distance returns zero radiance. CPU `transmittance` retains its incident-spectrum RGB convention; normalize it by the zero-distance white point before comparison with unit-normalized volume transmission. The browser suite reads back volume samples and compares both radiance and normalized RGB transmission with this reference, including the zero-distance identity.

## Absorption and diffuse higher orders

`createAtmosphereLayer` accepts an absorption-only medium with strictly ordered wavelength/cross-section samples and a reference molecular number density. Extinction is interpolated cross section times density; scattering is zero. `tent` profiles use `max(0, 1 - abs((height-referenceHeight)/scaleHeight))` inside their declared support. CPU and GPU pack the same analytic profile. Planet-specific cross sections and column assumptions belong to the consuming body composition, not to generic transport defaults.

`IAtmosphereSkyOptions.multipleScattering` defaults to false, preserving the direct single-scattering reference mode. When enabled, a 32 by 16 spectral table adds an isotropic higher-order closure following [Hillaire 2020](https://diglib.eg.org/items/8a3e5350-18b3-46bd-9274-3add5af88c75). It samples 32 deterministic sphere directions and 24 view segments at each altitude and Sun zenith, accumulates the isotropic source and scattering-return probability, then sums the geometric series. It includes no ground bounce. This is an approximation to multiple scattering, not a full angular radiative-transfer solution or an observational calibration.

The additional work is eight dispatches for 24 wavelengths, once per snapshot after solar-column construction. At most one higher-order dispatch is encoded per call, even if the requested budget is larger; hosts must submit and continue until `encode` returns true. It is not rebuilt on camera translation or rotation. View integration samples this shared diffuse source for both sky and finite-distance aerial transport. Public group 0 additionally binds the spectral table at 9 and its clamped filtering sampler at 10. `bindings().multipleRadiance` exposes it for numerical qualification when enabled. The browser harness compares table values against independent double-precision CPU quadrature using direct solar columns. The existing single-scattering CPU/GPU suite stays enabled separately.

### Orbital regression checkpoint

`tests/browser/orbital-aerial.js` compares native WGSL depth lookup and GPU transport with independent double-precision spectral integration. It covers pre-entry vacuum, partial paths, a complete approximately 101.5 km nadir air column, a 583.3 km grazing column, the same nadir ray from 1000 km and 10000 km observer altitude, and deep planetary shadow with single and multiple scattering. The representative complete nadir radiance error fell from about 75% to below 0.3%; the maximum absolute RGB radiance error across the current probes is below 0.00006 and normalized transmission error below 0.00073. These are numerical checks, not observational calibration or universal error guarantees.

`aerial-horizon.js` retains an independent finite-distance continuity check across the reference tangent. The current two-sided radiance jump is approximately 0.0000018. Babylon's surface reader has a separate native-image radial-profile regression in spacexr-rendering, because central-ray transport checks alone do not detect erroneous angular reprojection.

Local verification used the existing source browser harness without installing a packed consumer. The standard `test:webgpu` wrapper performs an npm installation internally and was not invoked under the owner's no-install policy. The package check, build and targeted GPU functions ran with already installed tools.

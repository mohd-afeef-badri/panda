
//-----------------------------------------------------------------------------
//
// Name       : square.geo
// Author     : Mohd Afeef BADRI
// Date       : 19 / Jan / 2026
//
// ----------------------------------------------------------------------------
// Comment    : simple square mesh
//
// Parameters : length          : this is the length of square
//              fillet_radius   : side fillet radius
//
// Usage      : gmsh square.geo -setnumber length 1.0 -setnumber fillet_radius 0  -2
//
//
//-----------------------------------------------------------------------------

DefineConstant[ length = {1.0, Min .0001, Max 100, Step 1,
                         Name "Parameters/ length"} ];

DefineConstant[ fillet_radius = {0.0, Min .0001, Max length/3., Step 1,
                         Name "Parameters/ fillet_radius"} ];

SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, length, length, fillet_radius};
Physical Surface("surface", 5) = {1};
Physical Curve("boundary", 6) = {4, 1, 2, 3};
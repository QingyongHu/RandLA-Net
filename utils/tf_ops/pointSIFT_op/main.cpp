#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;

REGISTER_OP("SelectionKRadius")
        .Attr("radius: float")
        .Input("idx: int32")
        .Input("val: float32")
        .Output("outi: int32")
        .Output("out: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle dims1;
            c->WithRank(c->input(0), 3, &dims1);
            ::tensorflow::shape_inference::ShapeHandle dims2;
            c->WithRank(c->input(1), 3, &dims2);
            c->set_output(0, c->input(0));
            c->set_output(1, c->input(0));
            return Status::OK();
        });
REGISTER_OP("CubeSelect")
        .Attr("radius: float")
        .Input("xyz: float32")
        .Output("idx: int32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle dim1;
            c->WithRank(c->input(0), 3, &dim1); // batch_size * npoint * 3
            ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dim1, 0), c->Dim(dim1, 1), 8});
            c->set_output(0, output1); // batch_size * npoint * 8
            return Status::OK();
        });
REGISTER_OP("CubeSelectTwo")
        .Attr("radius: float")
        .Input("xyz: float32")
        .Output("idx: int32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle dim1;
            c->WithRank(c->input(0), 3, &dim1); // batch_size * npoint * 3
            ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dim1, 0), c->Dim(dim1, 1), 16});
            c->set_output(0, output1); // batch_size * npoint * 16
            return Status::OK();
        });
REGISTER_OP("CubeSelectFour")
        .Attr("radius: float")
        .Input("xyz: float32")
        .Output("idx: int32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle dim1;
            c->WithRank(c->input(0), 3, &dim1); // batch_size * npoint * 3
            ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dim1, 0), c->Dim(dim1, 1), 32});
            c->set_output(0, output1); // batch_size * npoint * 32
            return Status::OK();
        });



void selectionKRadiusLauncher(int b, int m, int k, float radius, const int* idx, const float* val, int* idx_out, float* val_out);
class SelectionKRadiusOp: public OpKernel {
public:
    explicit SelectionKRadiusOp(OpKernelConstruction * context):OpKernel(context){
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
    }
    void Compute(OpKernelContext* context) override {
        const Tensor& idx_tensor = context->input(0);
        const Tensor& val_tensor = context->input(1);
        int b = idx_tensor.shape().dim_size(0);
        int m = idx_tensor.shape().dim_size(1);
        int k = idx_tensor.shape().dim_size(2);

        Tensor* idx_out_tensor = nullptr;
        Tensor* val_out_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,k}, &idx_out_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m,k}, &val_out_tensor));

        auto idx_flat = idx_tensor.flat<int>();
        auto val_flat = val_tensor.flat<float>();
        const int* idx = &(idx_flat(0));
        const float* val = &(val_flat(0));

        auto idx_out_flat = idx_out_tensor->flat<int>();
        auto val_out_flat = val_out_tensor->flat<float>();
        int* idx_out = &(idx_out_flat(0));
        float* val_out = &(val_out_flat(0));
        selectionKRadiusLauncher(b, m, k, radius_, idx, val, idx_out, val_out);
    }
private:
    float radius_;
};
REGISTER_KERNEL_BUILDER(Name("SelectionKRadius").Device(DEVICE_GPU),SelectionKRadiusOp);

void cubeSelectLauncher(int b, int n, float radius, const float* xyz, int* idx_out);
class CubeSelectOp : public OpKernel {
public:
    explicit CubeSelectOp(OpKernelConstruction * context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& xyz_tensor = context->input(0);
        int b = xyz_tensor.shape().dim_size(0);
        int n = xyz_tensor.shape().dim_size(1);

        Tensor* idx_out_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, 8}, &idx_out_tensor));
        auto xyz_flat = xyz_tensor.flat<float>();
        const float* xyz = &(xyz_flat(0));

        auto idx_out_flat = idx_out_tensor->flat<int>();
        int* idx_out = &(idx_out_flat(0));
        cubeSelectLauncher(b, n, radius_, xyz, idx_out);
    }
private:
    float radius_;
};
REGISTER_KERNEL_BUILDER(Name("CubeSelect").Device(DEVICE_GPU), CubeSelectOp);

void cubeSelectTwoLauncher(int b, int n, float radius, const float* xyz, int* idx_out);
class CubeSelectTwoOp : public OpKernel {
public:
    explicit CubeSelectTwoOp(OpKernelConstruction * context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& xyz_tensor = context->input(0);
        int b = xyz_tensor.shape().dim_size(0);
        int n = xyz_tensor.shape().dim_size(1);

        Tensor* idx_out_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, 16}, &idx_out_tensor));
        auto xyz_flat = xyz_tensor.flat<float>();
        const float* xyz = &(xyz_flat(0));

        auto idx_out_flat = idx_out_tensor->flat<int>();
        int* idx_out = &(idx_out_flat(0));
        cubeSelectTwoLauncher(b, n, radius_, xyz, idx_out);
    }
private:
    float radius_;
};
REGISTER_KERNEL_BUILDER(Name("CubeSelectTwo").Device(DEVICE_GPU), CubeSelectTwoOp);

void cubeSelectFourLauncher(int b, int n, float radius, const float* xyz, int* idx_out);
class CubeSelectFourOp : public OpKernel {
public:
    explicit CubeSelectFourOp(OpKernelConstruction * context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& xyz_tensor = context->input(0);
        int b = xyz_tensor.shape().dim_size(0);
        int n = xyz_tensor.shape().dim_size(1);

        Tensor* idx_out_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, 32}, &idx_out_tensor));
        auto xyz_flat = xyz_tensor.flat<float>();
        const float* xyz = &(xyz_flat(0));

        auto idx_out_flat = idx_out_tensor->flat<int>();
        int* idx_out = &(idx_out_flat(0));
        cubeSelectFourLauncher(b, n, radius_, xyz, idx_out);
    }
private:
    float radius_;
};
REGISTER_KERNEL_BUILDER(Name("CubeSelectFour").Device(DEVICE_GPU), CubeSelectFourOp);

'''
Code is borrowed from vqvae https://github.com/ritheshkumar95/pytorch-vqvae
See original paper in https://arxiv.org/pdf/1711.00937
'''

import torch
from torch.autograd import Function
import pdb

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')
    
class HierarchicalResidualVectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebooks):
        with torch.no_grad():
            indices = []
            inputs_size = inputs.size()
            embedding_size = codebooks[0].weight.size(1)
            inputs_flatten = inputs.view(-1, embedding_size)
            for i,cb in enumerate(codebooks):
                codebook = cb.weight
                codebook_sqr = torch.sum(codebook ** 2, dim=1)
                inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

                # Compute the distances to the codebook
                distances = torch.addmm(codebook_sqr + inputs_sqr,
                    inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
                _, indices_flatten = torch.min(distances, dim=1)
                # if i == 0:
                #     _, indices_flatten = torch.min(distances, dim=1)
                # else:
                #     distances = distances.view(inputs_flatten.size(0),codebooks[i-1].weight.size(0),-1)
                #     _, indices_flatten = torch.min(distances, dim=2)
                #     indices_flatten = torch.gather(indices_flatten, dim=1, index=indices[-1].view(-1,1)).view(-1) + indices[-1]*codebooks[0].weight.size(0)

                # residual
                inputs_flatten -= torch.index_select(codebook, dim=0, index=indices_flatten).detach()
                ## no residual
                indices.append(indices_flatten)
                
            for i,idc in enumerate(indices):
                indices[i] = idc.view(*inputs_size[:-1])
                ctx.mark_non_differentiable(indices[i])

            return indices
    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)
    
class HierarchicalResidualVectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook, indices):
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)
    
    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook,None)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
hrvq = HierarchicalResidualVectorQuantization.apply
hrvq_st = HierarchicalResidualVectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st, hrvq, hrvq_st]

classdef Replicate < dagnn.ElementWise
    properties 
        childIds = [];
        parentIds = [];
    end

    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs)==1); 
            outputs{1} = inputs{1}(:,:,obj.parentIds,:);
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputSlices = cellfun(@(I)(sum(derOutputs{1}(:,:,I,:),3)),...
                                     obj.childIds, 'UniformOutput', false);
            derInputs{1} = cat(3, derInputSlices{:});
            derParams = {};
        end

        function outputSizes = getOutputSizes(obj, inputSizes)
            np = numel(obj.childIds);
            nc = numel(obj.parentIds);
            isz = inputSizes{1};
            assert(isz(3) == np); 
            outputSizes{1} = [isz(1),isz(2),nc,isz(4)];
        end

        function rfs = getReceptiveFields(obj)
            numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
            rfs.size = [1 1] ;
            rfs.stride = [1 1] ;
            rfs.offset = [1 1] ;
            rfs = repmat(rfs, numInputs, 1) ;
        end

        function obj = Replicate(varargin)
            obj.load(varargin) ;
        end
    end
end

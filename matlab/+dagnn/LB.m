classdef LB < dagnn.ElementWise
    properties (Transient)
        average = 0
        numAveraged = 0
    end

    methods
        function outputs = forward(obj, inputs, params)
            KLD = inputs{1};
            NLL = inputs{2};
            X = inputs{3};
            LB = - KLD - NLL; 
            outputs{1} = LB;
            n = obj.numAveraged ;
            m = n + size(X,4);
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = [];
            derInputs{2} = [];
            derParams = {} ;
        end

        function reset(obj)
            obj.average = 0 ;
            obj.numAveraged = 0 ;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes{1} = inputSizes{1} ;
        end

        function rfs = getReceptiveFields(obj)
            numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
            rfs.size = [1 1] ;
            rfs.stride = [1 1] ;
            rfs.offset = [1 1] ;
            rfs = repmat(rfs, numInputs, 1) ;
        end

        function obj = LB(varargin)
            obj.load(varargin) ;
        end
    end
end

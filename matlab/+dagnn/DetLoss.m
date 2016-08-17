classdef DetLoss < dagnn.ElementWise
    properties
        loss = 'logistic'
        opts = {}
    end
    properties (Transient)
        average = 0
        numAveraged = 0
        
        instanceWeights = []
        sampleSize = 0
    end

    methods
        function outputs = forward(obj, inputs, params)
            X = inputs{1}; 
            c = inputs{2}; 
            obj.instanceWeights = cast(c ~= 0, 'like', c) ;
            obj.sampleSize = sum(obj.instanceWeights(:));

            a = -c.*X ;
            b = max(0, a) ;
            t = b + log(exp(-b) + exp(a-b)) ;

            Y = obj.instanceWeights(:)' * t(:) ;
            %outputs{1} = Y / obj.sampleSize;
            outputs{1} = Y;
            
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            X = inputs{1};
            c = inputs{2}; 
            dzdy = derOutputs{1};

            dzdy = dzdy * obj.instanceWeights;
            Y = - dzdy .* c ./ (1 + exp(c.*X)) ;
            
            %derInputs{1} = Y / obj.sampleSize;
            derInputs{1} = Y; 
            derInputs{2} = [] ;
            derParams = {} ;
        end

        function reset(obj)
            obj.average = 0 ;
            obj.numAveraged = 0 ;
        end

        function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
            outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
        end

        function rfs = getReceptiveFields(obj)
        % the receptive field depends on the dimension of the variables
        % which is not known until the network is run
            rfs(1,1).size = [NaN NaN] ;
            rfs(1,1).stride = [NaN NaN] ;
            rfs(1,1).offset = [NaN NaN] ;
            rfs(2,1) = rfs(1,1) ;
            rfs(3,1) = rfs(1,1) ;
        end

        function obj = DetLoss(varargin)
            obj.load(varargin) ;
        end
    end
end

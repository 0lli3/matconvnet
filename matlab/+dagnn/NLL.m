classdef NLL < dagnn.ElementWise
    properties
        opts = {}
    end

    properties (Transient)
        average = 0
        numAveraged = 0
    end

    methods
        function outputs = forward(obj, inputs, params)
            Y = squeeze(inputs{1}); 
            X = squeeze(inputs{2});
            NLL = - sum(sum(X.*log(Y+1e-12) + (1-X).*log(1-Y+1e-12)));
            outputs{1} = NLL;
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            Y = inputs{1};
            X = inputs{2};
            dLdY = - (X./(Y+1e-12) - (1-X)./(1-Y+1e-12));
            %dLdY = - (Y - X) ./ (Y.*(1-Y+1e-12) + 1e-12); 
            derInputs{1} = dLdY;
            derInputs{2} = [];
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
        end

        function obj = NLL(varargin)
            obj.load(varargin) ;
        end
    end
end

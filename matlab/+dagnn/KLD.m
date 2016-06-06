classdef KLD < dagnn.ElementWise
    properties
        opts = {}
    end

    properties (Transient)
        average = 0
        numAveraged = 0
    end

    methods
        function outputs = forward(obj, inputs, params)
            mu = squeeze(inputs{1});
            logvar = squeeze(inputs{2});
            sig = exp(logvar/2);
            KLD = -1/2 * sum(sum(1+log(sig.^2)-mu.^2-sig.^2));
            outputs{1} = KLD;
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            mu = inputs{1};
            logvar = inputs{2};
            dLdmu = mu;
            % BUG 
            %dLdlogvar = -(1./sig - sig);
            dLdlogvar = -(1 - exp(logvar))/2;
            derInputs{1} = derOutputs{1} .* dLdmu;
            derInputs{2} = derOutputs{1} .* dLdlogvar; 
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

        function obj = KLD(varargin)
            obj.load(varargin) ;
        end
    end
end

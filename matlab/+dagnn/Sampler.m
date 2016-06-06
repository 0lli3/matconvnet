classdef Sampler < dagnn.ElementWise
    properties
        eps
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            mu = inputs{1};
            logvar = inputs{2};
            sig = exp(logvar/2);
            obj.eps = randn(size(mu));
            z = mu + sig.*obj.eps;
            outputs{1} = z; 
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            logvar = inputs{2};
            sig = exp(logvar/2);
            derInputs{1} = derOutputs{1} ;
            %% BUG (input is not sigma, instead it is logvar) 
            % derInputs{2} = derOutputs{1} ;
            derInputs{2} = derOutputs{1}.*(obj.eps.*sig/2);
            derParams = {} ;
        end
    end
end

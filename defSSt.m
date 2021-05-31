%Simple undersampling operator
function [ S,St ] = defSSt( ind,res )

    function y = Sdef(x,ind)
        y = x(ind);
    end
    function y = Stdef(x,ind,res)
        y = zeros(res);
        y(ind) = x;
    end

    S = @(x) Sdef(x,ind);
    St = @(x) Stdef(x,ind,res);
end


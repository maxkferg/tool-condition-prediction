function xthreshold = thresholdIntersection(x,y,threshold)
% thresholdIntersection. Return the points where line (x,y) crosses @threshold
    %@x and @y should be column vectors of points
    %Threshold should be a single value
    if (length(x)~=length(y))
        throw('x and y must be the same length')
    end
    
    xthreshold = [];
    for i=2:length(x)
        y1 = y(i-1) - threshold;
        y2 = y(i) - threshold;
        if (y1*y2<0)
            x1 = x(i);
            x2 = x(i-1);
            xthreshold(end+1) = x1+ (x2-x1) * abs(y1/(y2-y1));
        end
    end

end


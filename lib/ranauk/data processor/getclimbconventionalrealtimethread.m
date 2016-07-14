%Code to extract which side of the tool cut more elements in order to
%classify as climb, conventional or both

% Input : Xnew, Xprev, Ynew, Yprev, elements, slope, pslope, middle
%Output : T,B

if Xnew > Xprev && Ynew > Yprev
    if Cy(e) - slope*Cx(e) + middle <0
        B = B+1;
    else
        T = T+1;
    end
end

if Xnew < Xprev && Ynew > Yprev
    if Cy(e) - slope*Cx(e) + middle >0
        T = T+1;
    else
        B = B+1;
    end
end

if Xnew < Xprev && Ynew < Yprev
    if Cy(e) - slope*Cx(e) + middle >0
        T = T+1;
    else
        B = B+1;
    end
end

if Xnew > Xprev && Ynew < Yprev
    if Cy(e) - slope*Cx(e) + middle >0
        B = B+1;
    else
        T = T+1;
    end
end

if Xnew == Xprev
    if Ynew>Yprev
        if Cx(e)> Xprev
            T = T+1;
        else
            B = B+1;
        end
    end
    if Yprev>Ynew
        if Cx(e)> Xprev
            B = B+1;
        else
            T = T+1;
        end
    end
end

if Ynew == Yprev
    if Xnew > Xprev
        if Cy(e) > Yprev
            B = B+1;
        else
            T = T+1;
        end
    end
    if Xprev > Xnew
        if Cy(e) > Yprev
            T = T+1;
        else
            B = B+1;
        end
    end
end

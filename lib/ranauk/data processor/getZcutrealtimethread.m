%For getting depths of cut during plunging

%Input : Elements, Xnew, Ynew, r, Zvalue, currentZ(e)
%Output : D(e), updated currentZ(e), IdX, IdY, IAoC, LoC, TLoC

for e = 1:N
    if (Cx(e)-Xnew)^2 + (Cy(e)-Ynew)^2 - r^2 <= 0
        if currentZ(e) > Zvalue
            D(e)=abs(currentZ(e)-Zvalue);
            currentZ(e)=Zvalue;
        end
    end
end

if isnan(mode(D(find(D>0))')) == 1
        Depth = 0;
else
        Depth= mode(D(find(D>0))');
end

IdX = 0; %IdX
IdY = 0; %IdY
IAoC = pi*r^2; %Area
LoC=Depth;
TLoC = Depth;%Total Length of cut

%These variables are written into EData separately outside the loop!!!
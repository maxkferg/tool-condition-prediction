%getcircle
C = zeros(2,3); tnew = zeros(1,2);tprev = zeros(1,2);
%given 2 points and radius (R), find two possible centers of the circular
d = EData{i,17}; Mx = .5*(Xnew+Xprev); My = .5*(Ynew+Yprev); %distance, midpt between Prev,New

%store in (x,y; x,y;) format
C(1,1) = Mx + sqrt(R^2-(d/2)^2)*-1*EData{i,16}/d;
C(1,2) = My + sqrt(R^2-(d/2)^2)*EData{i,15}/d;
C(2,1) = Mx - sqrt(R^2-(d/2)^2)*-1*EData{i,16}/d;
C(2,2) = My + sqrt(R^2-(d/2)^2)*EData{i,15}/d;

if isreal(C(1,1))==1

for j = 1:2
    tnew(j) = atan((Ynew-C(j,2))/(Xnew-C(j,1)));
    tprev(j) = atan((Yprev-C(j,2))/(Xprev-C(j,1)));
    C(j,3) = tnew(j) - tprev(j); 
end
C = sortrows(C,3); %Sort possible center points based on angle (Prev,C,New) (ascending)
tnew=sort(tnew);tprev=sort(tprev);
% If angle is negative that indicates a G02 cut -> C = C(1,:)
if sum(strcmp(EData(i,:),'G02')) == 1; %Clockwise circle
    X = C(1,1); Y = C(1,2); tnew = tnew(1); tprev = tprev(1);
elseif sum(strcmp(EData(i,:),'G03')) == 1;%Counterclockwise circle
    Y = C(2,1); Y = C(2,2); tnew = tnew(2); tprev = tprev(2);
end
%cut
for e = 1:N
    if (Cx(e)-X)^2 + (Cy(e)-Y)^2 - (R+r)^2 <= 0    
        if (Cx(e)-X)^2 + (Cy(e)-Y)^2 - (R-r)^2 >= 0
            t1 = atan((Cy(e)-Y)/(Cx(e)-X)) - tprev;
            t2 = tnew - atan((Cy(e)-Y)/(Cx(e)-X));
           if t1>0 && t2 < 0
                    in(e) = in(e)+1;
           end
        end
    end
    
    if in(e) > 0
        if currentZ(e)>Zvalue
            if sum(strcmp(EData(i,:),'G02')) == 1
                if (Cx(e)-X)^2 + (Cy(e)-Y)^2 - (R)^2 > 0
                    T = T+1;
                elseif (Cx(e)-X)^2 + (Cy(e)-Y)^2 - (R)^2 < 0
                    B = B+1;
                end
            elseif sum(strcmp(EData(i,:),'G03')) == 1
                if (Cx(e)-X)^2 + (Cy(e)-Y)^2 - (R)^2 > 0
                    B = B+1;
                elseif (Cx(e)-X)^2 + (Cy(e)-Y)^2 - (R)^2 < 0
                    T = T+1;
                end
            end
cut2(count2+1) = e; %If elements were cut to a new depth        
count2 = count2+1; %Counter for flag for area of cut

        end
D(e)=abs(Zvalue-currentZ(e));
currentZ(e)=Zvalue;
    end
    
end

if count2 >0
    flag =1; %If elements were cut in this block, then flag = 1 -> calculate IAoC, IdX, IdY
end

else
    cut2 = 0; count2=0;D = 0;T = 0; B = 0;
end   

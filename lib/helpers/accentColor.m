function color = accentColor(i)
    % Return the correct accent color for the NIST presentation
    % Dark colors
    colors = [
        140,  60,  35;
          7,  80,  91;
         82,  41,  78;
         27,  94,  84;
         77,  79,  83;
        210, 193, 151;
    ];
    % Bright Colors
    colors = [
          0, 152, 219;
          0, 155, 118;
         233, 131,  0;
          83, 40,  79;
         27,  94,  84;
         77,  79,  83;
        210, 193, 151;
        140, 21,   21; % 7 Cardinal red
    ];
    color = colors(i,:)/256;
end

dir="$1"
dirout="$2"
shopt -s nullglob

# cat by YYYYMM
for f in "$dir"/wind_202[0-9][0-9]*.grib; do
    fname=$(basename "$f")
    ym=${fname:5:6}        # extract YYYYMM
    files["$ym"]+="$f "
done

for ym in "${!files[@]}"; do
    cat ${files[$ym]} > "$dirout/wind_${ym:0:4}_${ym:4:2}.grib"
done

# repeat for precip files
for f in "$dir"/precip_202[0-9][0-9]*.grib; do
    fname2=$(basename "$f")
    ym2=${fname2:7:6}        # extract YYYYMM
    files2["$ym2"]+="$f "
done

for ym2 in "${!files2[@]}"; do
    cat ${files2[$ym2]} > "$dirout/precip_${ym2:0:4}_${ym2:4:2}.grib"
done

cp "$dir"/*_202[0-9]_*.grib "$dirout"/
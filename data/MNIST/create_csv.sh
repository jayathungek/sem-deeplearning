root_dir="$1"
csv_file="$root_dir/mnist_dataset.csv"
re='^[0-9]+$'

echo "mapping files in directory: $root_dir"
echo "image_path, label" > "$csv_file"
root_to_write=$(basename -a "$root_dir")

while read -r label; do 
  if [[ $label =~ $re ]] ; then
    while read -r image; do
      echo "$root_to_write/$label/$image, $label" >> "$csv_file"
    done < <(ls "$root_dir/$label")
    echo "$label: $(ls "$root_dir/$label" | wc -l)"
  fi
  
done < <(ls "$root_dir")

#!/bin/bash

launcher_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

"$launcher_dir/tools/sona_launcher.sh" start
launcher_status=$?

if [ "$launcher_status" -ne 0 ]; then
    echo
    read -r -p "Sona could not start. Press Return to close this window. " _
fi

exit "$launcher_status"

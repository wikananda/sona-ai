#!/bin/bash

launcher_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

"$launcher_dir/tools/sona_launcher.sh" stop
launcher_status=$?

if [ "$launcher_status" -ne 0 ]; then
    echo
    read -r -p "Sona could not be stopped cleanly. Press Return to close this window. " _
else
    sleep 1
fi

exit "$launcher_status"

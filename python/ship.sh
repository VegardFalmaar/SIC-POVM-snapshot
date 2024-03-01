# script for shipping results to and from Betzy

# it is designed to be run from the development computer, NOT FROM BETZY

# requires that Betzy is defined in ~/.ssh/config as 'betzy' such that one may
# log in to betzy with the command 'ssh betzy'

# it syncs the contents of the directories
# ${HOME}/SIC-POVM-results
# on cluster and dev computer using rsync

# files that exist both on the source and destination computer will be
# overwritten on the destination computer with the values they have on the
# source computer


### parameters passed to rsync
# -a: archive mode, ensures that symbolic links, devices, attributes,
#   permissions, ownerships, etc. are preserved in the transfer
# -v: verbose
# -z: use compression to reduce the file size of the transfer
# --delete: delete extraneous files from the destination


### how to run

# bash ship.sh cluster
#   to ship results from current computer to cluster

# bash ship.sh home
#   to ship results from cluster to current computer

if [ "${1}" == "cluster" ]; then
    echo "Setting up shipping to cluster. The following changes will be made."

    # transfer the contents of the folder
    # ${HOME}/SIC-POVM-results
    # on the local machine into the folder
    # ${HOME}/SIC-POVM-results
    # on Betzy
    rsync -avz --delete ${HOME}/SIC-POVM-results/ betzy:SIC-POVM-results --dry-run

    read -p "Continue? Type 'yes' to confirm: " ansr
    if [[ $ansr == "yes" ]]; then
        rsync -avz --delete ${HOME}/SIC-POVM-results/ betzy:SIC-POVM-results
    fi
elif [ "${1}" == "home" ]; then
    echo "Setting up shipping from cluster. The following changes will be made."

    # transfer the contents of the folder
    # ${HOME}/SIC-POVM-results
    # on Betzy
    # into the folder
    # ${HOME}/SIC-POVM-results
    # on the local machine
    rsync -avz --delete betzy:SIC-POVM-results/ ${HOME}/SIC-POVM-results --dry-run

    read -p "Continue? Type 'yes' to confirm: " ansr
    if [[ $ansr == "yes" ]]; then
        rsync -avz --delete betzy:SIC-POVM-results/ ${HOME}/SIC-POVM-results
    fi
else
    echo Unknown option 1: $1
fi

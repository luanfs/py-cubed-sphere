#!/bin/bash
# P. Peixoto - Jul 2012
# modified by Luan Santos - 2022

date=` date +%F `
version=` date +%y.%m.%d `

#output="py-dimsplit$version.tar.bz2"
output="py-cs.tar.bz2"
bkpdir="py-cs" 

#-------------------------------------------------------------------------------------------------------
# remote host 1 - ime.usp.br
user_remote_host1="luansantos"
remote_host1="brucutuiv.ime.usp.br"
remote_host1_dir="/var/tmp"
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
# remote host 2 - ybytu
user_remote_host2="luansantos"
remote_host2="ybytu"
remote_host2_dir="doc/py-cs"
#-------------------------------------------------------------------------------------------------------


#Edit place to sync relative to system used
dropdir="/home/luan/Dropbox/doc/code/"$bkpdir
echo "Sync with Dropbox:"
rsync -v -t -u $output  "$dropdir/."
echo "Synchronized with Dropbox"
echo

#remote server remote_host1 backup sync
echo "Sending to $remote_host1:"
rsync -t -v -z -a --progress $output $user_remote_host1@$remote_host1:$remote_host1_dir
echo "Sent to $user_remote_host1@$remote_host1"
echo

#remote server remote_host2 backup sync
echo "Sending to $remote_host2:"
ssh -t $user_remote_host1@$remote_host1 "rsync -t -v -z -a --progress $remote_host1_dir/$output $user_remote_host2@$remote_host2:$remote_host2_dir; rm -rf $output"
echo "Sent to $user_remote_host2@$remote_host2"
echo

#-------------------------------------------------------------------------------------------------------
# untar
echo "Untar at $remote_host2:"
ssh -t $user_remote_host1@$remote_host1 "ssh -t $user_remote_host2@$remote_host2 <<EOF
	cd $remote_host2_dir;
	tar -xvf $output;
	rm -rf $output;
EOF"
echo "Untar and compilation at $remote_host2 done."
#-------------------------------------------------------------------------------------------------------

# remove tar file
rm -rf $output

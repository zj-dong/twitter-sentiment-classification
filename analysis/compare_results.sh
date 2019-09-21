# compare the results from file

if [[ ! $# -eq 2 ]]
    then
        echo "Only comparision between 2 csv files"
        echo "Usage: ./compare_results.sh file1 file2"
        exit 0
fi

# perform a join on the first column
join -t ","  ../data/gt.csv $1 | join -t "," - $2 > comparison.csv


file1=$(awk -F',' '{if($2==$3) print $1}' comparison.csv | wc -l)
file2=$(awk -F',' '{if($2==$4) print $1}' comparison.csv | wc -l)
diff=$(awk -F',' '{if($3!=$4) print $1}' comparison.csv | wc -l)

# ids that file 1 classifies correct but not file 2
file1_exclusive=$(awk -F',' '{if(($2==$3) && ($2 != $4)) print $1}' comparison.csv | wc -l)

# ids that file 1 classifies correct but not file 2
file2_exclusive=$(awk -F',' '{if(($2!=$3) && ($2 == $4)) print $1}' comparison.csv | wc -l)

# ids that both file 1.2 classifies correctly
both=$(awk -F',' '{if(($2==$3) && ($2 == $4)) print $1}' comparison.csv | wc -l)

# ids that file 1 classifies correct but not file 2
neither=$(awk -F',' '{if(($2!=$3) && ($2 != $4)) print $1}' comparison.csv | wc -l)

echo $(cat gt.csv | wc -l) " tweets in total"
echo $file1 " tweets file 1 got right"
echo $file2 " tweets file 2 got right"

echo $diff "tweets are file 1,2 don't agree"
echo $file1_exclusive " tweets ONLY file 1 got right"
echo $file2_exclusive " tweets ONLY file 2 got right"
echo $both " tweets BOTH files got right"
echo $neither " tweets BOTH files got wrong"

# The following function can be adapted to visualize the certain tweets each 2 csv are classified differently.
# groundtruth (id + tweets) 1.csv 2.csv
cat comparison.csv  | awk -F, ' BEGIN {
        print "ID, polarity A,polarity B, tweet"
} NF > 2 {
        if ( $3 != $4 )
                print $1, $3, $4, $2
} ' OFS=,

import argparse
import sys
import pandas as pd
from functools import reduce
from itertools import combinations
if __name__ == '__main__':

    # Instantiate the parser
    # parser = argparse.ArgumentParser()

    print("sys.argv", sys.argv)

    # if len(sys.argv) < 2:
    #     print("Error: Input at least one file!\nUsage: python compare_results.py 1.csv 2.csv ...")
    #     exit(0)
    #
    # num_file = len(sys.argv) - 1

    df = []
    df_name = ['id']
    for file in sys.argv[1:]:
    # for file in ['1.csv', '2.csv', '3.csv', '4.csv']:
        print(file)
        df.append(pd.read_csv(file))
        df_name.append(file.strip(".csv"))

    # finally append the groundtruth
    df.append(pd.read_csv('gt.csv'))
    df_name.append('gt')

    df_merged = reduce(lambda left, right: pd.merge(left, right, on='id'), df)
    # have to rename each column
    df_merged.columns = df_name
    print(df_merged)

    # pairwise comparison between each files
    pairwise_iter = list(combinations(df_name[1:-1],2))

    # (df_merged.1 == df_merged.2).value_counts()
    for pair in pairwise_iter:

        # only one of the file got correct
        # print((df_merged[pair[0]] != df_merged[pair[1]]) & (df_merged[pair[0]] == df_merged['gt']))

        left_exclusive = sum((df_merged[pair[0]] != df_merged[pair[1]]) & (df_merged[pair[0]] == df_merged['gt']))
        right_exclusive = sum((df_merged[pair[0]] != df_merged[pair[1]]) & (df_merged[pair[1]] == df_merged['gt']))
        both = sum((df_merged[pair[0]] == df_merged[pair[1]]) & (df_merged[pair[0]] == df_merged['gt']))
        neither = sum((df_merged[pair[0]] == df_merged[pair[1]]) & (df_merged[pair[0]] != df_merged['gt']))

        print("====== Result between " + pair[0] + " " + pair[1] + "=======")
        print("{}% sentences only {} got right".format(left_exclusive / 200000.0 * 100, pair[0]))
        print("{}% sentences only {} got right".format(right_exclusive / 200000.0 * 100, pair[1]))
        print("{}% sentences both got right".format(both / 200000.0 * 100))
        print("{}% sentences neither got right".format(neither / 200000.0 * 100))

    #
    #
    #
    #
    # print(pd.merge(df1, df2.rename(columns={'id1': 'id'}), on='id', how='left'))
    #

    # args = parser.parse_args()
    # print(args)
    #
    # if args.pos_arg < 1:
    #     parser.error("pos_arg cannot be larger than 10")
    #

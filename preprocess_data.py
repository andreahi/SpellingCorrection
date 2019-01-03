import random
import re

count = 0
with open("raw_data/wiki.no.txt", encoding="utf-8") as f:
    with open('data.txt', 'w') as the_file:
        for e in f:
            max = random.randint(5, 50)
            pattern = re.compile("( [a-zA-Z0-9_\\-åøæ ,]{2,"+str(max)+"}[.?!])")
            match = pattern.findall(e)
            if match:
                #print(match.groups())
                for match in match:
                    the_file.write(match.lstrip() + "\n")
                    if 'µ' in match:
                        print(match)
                    if len(match) > 150:
                        print("big sentence: ", match)
                    count += 1

print("total number of lines: " + str(count))

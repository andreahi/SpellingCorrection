import re
pattern = re.compile("(^[a-zA-Z0-9_-åøæ ]{40,150}\\.)")

count = 0
with open("raw_data/wiki.no.txt", encoding="utf-8") as f:
    with open('data.txt', 'a') as the_file:
        for e in f:
            match = pattern.match(e)
            if match:
                #print(match.groups())
                for match in match.groups():
                    the_file.write(match + "\n")
                    if len(match) > 199:
                        print("big sentence: ", match)
                    count += 1

print("total number of lines: " + str(count))

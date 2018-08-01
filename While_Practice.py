"""
Quick while practice
"""
# count = 0
# goes = 0
# while count < 50:
#     new_num = int(input('>'))
#     count += new_num
#     goes += 1
# print(count, goes)


# while True:
#     new_num = int(input('>'))
#     if new_num == 5:
#         pass
#     else:
#         break
# print(new_num)


while True:
    line = input('> ')
    if line[0] == '#':
        continue
    if line.lower() == 'done':
        break
    print(line)
print('Done!')

# foo comment
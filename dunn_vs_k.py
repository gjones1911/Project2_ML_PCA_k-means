from ML_Visualizations import *


all_dunn_idx = list([.23726209549321142,        # 2
                     .19216396304176414,        # 3
                     .22773139366760664,        # 4
                     .21827817994722423,        # 5
                     .21025299603036127,        # 6
                     .20238179359874742,        # 7
                     .20998020902762618,        # 8
                     .22877910007481758,        # 9
                     .2185594139393297,         # 10
                     .22249576088315665,        # 11
                     .2573433171687043,         # 12
                     .255078090760439,          # 13
                     .2545255025307531,         # 14`
                     .26276809006476476,        # 15
                     .2675752206323636,         # 16
                     .28027227805207183,        # 17
                     .27688933858004977,        # 18
                     .29303639470985576,        # 19
                     .29966305188741554,        # 20
                     .2949194153142218,         # 21
                     .3197205091508887,         # 22
                     .3325181494382632,        # 23
                     .3289108890362078])

orig_dunn_idx = list([.28,          # 2
                      .27,          # 3
                      .26,          # 4
                      .27,          # 5
                      .27,          # 6
                      .27,          # 7
                      .28,          # 8
                      .28,          # 9
                      .28,          # 10
                      .29,          # 11
                      .29,          # 12
                      .29,          # 13
                      .29,          # 14
                      .30,          # 15
                      .30,          # 16
                      .30,          # 17
                      .30,          # 18
                      .31,          # 19
                      .32,          # 20
                      .32,          # 21
                      .32,          # 22
                      .32,          # 23
                      .33])

pcak_dunn_idx = list([.11,          # 2
                      .09,          # 3
                      .11,          # 4
                      .13,          # 5
                      .15,          # 6
                      .16,          # 7
                      .18,          # 8
                      .19,          # 9
                      .20,          # 10
                      .22,          # 11
                      .23,          # 12
                      .24,          # 13
                      .25,          # 14
                      .26,          # 15
                      .26,          # 16
                      .26,          # 17
                      .26,          # 18
                      .28,          # 19
                      .29,          # 20
                      .29,          # 21
                      .28,          # 22
                      .30,          # 23
                      .31])


k_vals = list(range(2, 25))

print(k_vals)
print(all_dunn_idx)

fig = plt.figure(1)

plt.title('K value vs. Average Dunn index')
plt.xlabel('K value')
plt.ylabel('Average Dunn index')

plt.plot(k_vals, all_dunn_idx, 'r-',  linewidth=2)
plt.plot(k_vals, orig_dunn_idx, 'm-',  linewidth=2)
plt.plot(k_vals, pcak_dunn_idx, 'g-',  linewidth=2)

leg = plt.legend(['Overall dunn index', 'Original Data Dunn Index', 'K PC\'s Dunn index'], loc='best',
                 borderpad=0.3, shadow=False,
                 prop=matplotlib.font_manager.FontProperties(size='medium'),
                 markerscale=0.4)

leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)

plt.show()


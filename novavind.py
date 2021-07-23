import math
import collections
import sys
import os


def plotvetraki(v, min_teta, max_teta, r_bliz, r_vdal, weght, centr, save_dir):

    out = [['_']*weght for _ in range(weght)]
    for r in range(int(r_bliz), min(weght-1,int(r_vdal))):
        x = int(centr[0] +  r * math.cos(math.radians(min_teta)))
        y = int(centr[1] +  r * math.sin(math.radians(min_teta)))
        if x < weght-1 and y < weght-1 and x > 0 and y > 0:
            out[x][y] = '*'

        x = int(centr[0] +  r * math.cos(math.radians(max_teta)))
        y = int(centr[1] +  r * math.sin(math.radians(max_teta)))
        if x < weght-1 and y < weght-1 and x > 0 and y > 0:
            out[x][y] = '*'
    for n_fi in range(10):
        teta = min_teta + n_fi*(max_teta-min_teta)/10
        x = int(centr[0] +  int(r_bliz) * math.cos(math.radians(teta)))
        y = int(centr[1] +  int(r_bliz) * math.sin(math.radians(teta)))
        if x < weght-1 and y < weght-1 and x > 0 and y > 0:
            out[x][y] = '*'


    out[int(centr[0])][int(centr[1])] = 'X'
    for i, (x,y) in enumerate(v):
        out[int(x)][int(y)] = str(i)
    res = ''
    for i in range(weght):
        res += str(i)+''.join(out[i])+'\n'
    original_stdout = sys.stdout
    with open(os.path.join(save_dir, 'carta1.txt'), 'w') as f:
        sys.stdout = f
        print(res)
        sys.stdout = original_stdout

def dec_to_polar(v, centr):
    centr_vetr = [[x-centr[0], y-centr[1]] for x, y in v]
    # print(centr_vetr)
    polar = [[round(math.sqrt(x**2+y**2), 3), round(math.degrees(math.atan2(y, x)), 3)] for x, y in centr_vetr]
    return polar


def plot_one_photo(f1, out_dpi, out_dpi_vert, yam, ugol_obzora_hor, min_gamma, ugol_obzora_vert, h, save_dir):
    ee = [['_']*out_dpi for _ in range(out_dpi_vert)]
    min_teta = yam - ugol_obzora_hor // 2
    max_teta = yam + ugol_obzora_hor // 2
    dict = collections.defaultdict(list)
    dict2 = collections.defaultdict(list)

    outlist = []

    if min_teta < -180 or max_teta > 180:
        for k, v in f1.items():
            if max_teta > 180 and min_teta < 180:
                t1 = max_teta - 360
                t2 = -180
                t3 = 180
                t4 = min_teta
                if t2 <= v[1] <= t1:
                    s = (t1-v[1])/ugol_obzora_hor
                if t4 <= v[1] <= t3:
                    s = (ugol_obzora_hor - (v[1]-t4))/ugol_obzora_hor

            if max_teta > 180 and min_teta > 180:
                t2 = min_teta - 360
                s = (v[1]-t2) / ugol_obzora_hor
            if max_teta > -180 and min_teta < -180:
                t1 = max_teta
                t2 = -180
                t3 = 180
                t4 = 360+min_teta
                if t2 <= v[1] <= t1:
                    s = (t1 - v[1]) / ugol_obzora_hor
                if t4 <= v[1] <= t3:
                    s = (ugol_obzora_hor - (v[1] - t4)) / ugol_obzora_hor
            if max_teta < -180 and min_teta < -180:
                t1 = max_teta + 360
                s = (t1-v[1]) / ugol_obzora_hor

            dict[k].append((out_dpi - 1) * s)
    else:
        for k, v in f1.items():
            dict[k].append((out_dpi - 1) * (1 - (v[1] - min_teta) / ugol_obzora_hor))

    for k, v in f1.items():
        alfa = round(math.degrees(math.atan(v[0]/h)), 3)
        if 0.25*out_dpi < dict[k][0] < 0.75*out_dpi:
            s = (alfa - min_gamma)/ugol_obzora_vert
        else:
            s = (alfa - min_gamma)/ugol_obzora_vert

        dict2[k].append((out_dpi_vert-1) - (out_dpi_vert - 1) * s)

    for (k, v), (kv, vv) in zip(dict.items(), dict2.items()):
        ee[int(vv[0])][int(v[0])] = str(k)
        outlist.append((vv[0], v[0], str(k)))
        # print(vv[0], v[0])


    original_stdout = sys.stdout
    with open(os.path.join(save_dir, 'carta2.txt'), 'w') as f:
        sys.stdout = f
        print('\n'.join([''.join(x) for x in ee]))
        sys.stdout = original_stdout

    return outlist


def teorcart(yam, pitch, h, centrll, save_dir):
    weght = 100
    ugol_obzora_hor = 60
    ugol_obzora_vert = 70

    out_dpi = 60
    out_dpi_vert = 60

    all_vetr = []
    with open('./input/ves.txt', 'r') as file:
        sod = file.readlines()

    for row in sod:
        row = row.replace('\n', '')
        arr = row.split('\t')
        all_vetr.append([float(arr[0]), float(arr[1])])

    minx = min([x for x, y in all_vetr])
    maxx = max([x for x, y in all_vetr])
    miny = min([y for x, y in all_vetr])
    maxy = max([y for x, y in all_vetr])

    all_vetr = [[((weght - 1) * (x - minx) / (maxx - minx)), ((weght - 1) * (y - miny) / (maxy - miny))] for x, y
                in all_vetr]

    centr = (((weght - 1) * (centrll[0] - minx) / (maxx - minx)),
             ((weght - 1) * (centrll[1] - miny) / (maxy - miny)))

    polar = dec_to_polar(all_vetr, centr)

    min_teta = yam - ugol_obzora_hor // 2
    max_teta = yam + ugol_obzora_hor // 2

    min_gamma = max(pitch - ugol_obzora_vert // 2, 0)
    max_gamma = pitch + ugol_obzora_vert // 2

    r_bliz = h * math.tan(math.radians(min_gamma))
    r_vdal = h * math.tan(math.radians(max_gamma))
    # print(r_bliz, r_vdal)
    if r_vdal < 0:
        r_vdal = 10**9

    # print(centr)
    plotvetraki(all_vetr, min_teta, max_teta, r_bliz, r_vdal, weght, centr, save_dir)

    if not (min_teta < -180 or max_teta > 180):
        filtr = {i: [r, teta] for i, (r, teta) in enumerate(polar)
                 if (min_teta <= teta <= max_teta) and (r_bliz <= r <= r_vdal)}
    else:
        if max_teta > 180 and min_teta < 180:
            t1 = max_teta - 360
            t2 = -180
            t3 = 180
            t4 = min_teta
            filtr = {i: [r, teta] for i, (r, teta) in enumerate(polar)
                     if ((t2 <= teta <= t1) or (t4 <= teta <= t3)) and (r_bliz <= r <= r_vdal)}
        if max_teta > 180 and min_teta > 180:
            t1 = max_teta - 360
            t2 = min_teta - 360
            filtr = {i: [r, teta] for i, (r, teta) in enumerate(polar)
                     if (t2 <= teta <= t1) and (r_bliz <= r <= r_vdal)}
        if max_teta > -180 and min_teta < -180:
            t1 = max_teta
            t2 = -180
            t3 = 180
            t4 = 360+min_teta
            filtr = {i: [r, teta] for i, (r, teta) in enumerate(polar)
                     if ((t2 <= teta <= t1) or (t4 <= teta <= t3)) and (r_bliz <= r <= r_vdal)}
        if max_teta < -180 and min_teta < -180:
            t1 = max_teta + 360
            t2 = min_teta + 360
            filtr = {i: [r, teta] for i, (r, teta) in enumerate(polar)
                     if (t2 <= teta <= t1) and (r_bliz <= r <= r_vdal)}

    f1 = dict(sorted(filtr.items(), key=lambda item: item[1][1]))

    out = plot_one_photo(f1, out_dpi, out_dpi_vert, yam, ugol_obzora_hor, min_gamma, ugol_obzora_vert, h, save_dir)
    # print(out)
    return out
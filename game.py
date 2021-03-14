# -*- coding: utf-8 -*-

import numpy as np
import time
import json

file = open('data/data.json')
data = json.load(file)
diag4 = data['diag4']
diag5 = data['diag5']
diag6 = data['diag6']
diag7 = data['diag7']
diag8 = data['diag8']
lc1 = data['lc1']
lc2 = data['lc2']
lc3 = data['lc3']
lc4 = data['lc4']
corn = data['corn']
parity = data['parity']
openingBook = data['openingBook']

Diag_table = ((0x1020408000000000, 0x1010101), (0x80402010, 0x101010100000000),
              (0x804020100000000, 0x80200802), (0x1020408, 0x208208),
              (0x810204080000000, 0x101010101), (0x8040201008, 0x101010101000000),
              (0x8040201008, 0x20080200802), (0x102040810, 0x82082080000000),
              (0x408102040800000, 0x10101010101), (0x804020100804, 0x101010101010000),
              (0x10204081020, 0x20820820800000), (0x201008040201, 0x2008020080200),
              (0x204081020408000, 0x1010101010101), (0x80402010080402, 0x101010101010100),
              (0x1020408102040, 0x8208208208000), (0x100804020100804, 0x1004010040100401),
              (0x8040201008040201, 0x101010101010101), (0x102040810204080, 0x101010101010101))
column_table = ((0x8080808080808080, 0x2040810204081), (0x4040404040404040, 0x4081020408102),
                (0x2020202020202020, 0x8102040810204), (0x1010101010101010, 0x10204081020408),
                (0x808080808080808, 0x20408102040810), (0x404040404040404, 0x40810204081020),
                (0x202020202020202, 0x81020408102040), (0x101010101010101, 0x102040810204080))
corn_table = ((0x8080808000000000, 0x10204081), (0x4040404000000000, 0x10204081),
              (0x80808080, 0x10204081), (0x40404040, 0x10204081),
              (0xf0f0, 0x1001), (0xf0f, 0x11),
              (0x1010101, 0x10204081), (0x2020202, 0x10204081),
              (0x202020200000000, 0x10204081), (0x101010100000000, 0x10204081),
              (0xf0f0000000000000, 0x11), (0xf0f0000000000000, 0x1001))
form_2_to_3 = [int(bin(i)[2:], 3) for i in range(1024)]
HASH_TABLE_SIZE = 1 << 20
HASH_TABLE_MASK = HASH_TABLE_SIZE - 1
DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))  # 方向向量
BND = 1 << 48
mod_64 = 0xffffffffffffffff + 1
stageTable = [0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
              4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
              7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9,
              10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12]


class AI(object):
    direction = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    DIR_d = (8, -8, 1, -1, 9, -7, -9, 7)

    def __init__(self, chessboard_size, color, time_out=6):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.trans_table2 = {}

    def custom_ffs(self, x):
        if x == 0:
            return 0
        num = 1
        if (x & 0xffffffff) == 0:
            num += 32
            x >>= 32
        if (x & 0xffff) == 0:
            num += 16
            x >>= 16
        if (x & 0xff) == 0:
            num += 8
            x >>= 8
        if (x & 0xf) == 0:
            num += 4
            x >>= 4
        if (x & 0x3) == 0:
            num += 2
            x >>= 2
        if (x & 0x1) == 0:
            num += 1
        return num

    def evaluation(self, cur, opp):
        moves_ = self.count(self.get_move(cur, opp))
        moves_2 = self.count(self.get_move(opp, cur))
        table_value = ((12, 0.5, -6, -0.2), (10, 0.5, -5, 0.2), (3, 1, 0, 0))
        table_s = (0x8100000000000081, 0x7e8181818181817e, 0x0042000000004200, 0x003c7e7e7e7e3c00)
        brd = cur | opp
        total = self.count(brd)
        if total <= 40:
            stage = 0
        elif total <= 64 - 7:
            stage = 1
        else:
            stage = 2
        result = 0
        result += moves_2 - moves_
        # brd_stable = self.get_stable(brd)
        result += self.count(self.get_stable(cur)) - self.count(self.get_stable(opp))
        brd_front = self.get_front(brd)
        result += self.count(brd_front & cur) - self.count(brd_front & opp)

        for i in range(4):
            result += (self.count(cur & table_s[i]) - self.count(opp & table_s[i])) * \
                      table_value[stage][i]
        return result

    def change_rev(self, b, w, color):
        board = np.asarray([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]])
        for i in range(64):
            if b & 1:
                board[int((63 - i) / 8)][(63 - i) % 8] = color
            elif w & 1:
                board[int((63 - i) / 8)][(63 - i) % 8] = -color
            b >>= 1
            w >>= 1
        return board

    def get_front(self, brd: int) -> int:
        brd_reverse = ~brd
        brd_front = 0
        brd_front |= brd & (brd_reverse >> 1) & 0x7f7f7f7f7f7f7f7f
        brd_front |= brd & (brd_reverse << 1) & 0xfefefefefefefefe
        brd_front |= brd & (brd_reverse >> 8)
        brd_front |= brd & (brd_reverse << 8)
        brd_front |= brd & (brd_reverse >> 9) & 0x7f7f7f7f7f7f7f7f
        brd_front |= brd & (brd_reverse >> 7) & 0xfefefefefefefefe
        brd_front |= brd & (brd_reverse << 7) & 0x7f7f7f7f7f7f7f7f
        brd_front |= brd & (brd_reverse << 9) & 0xfefefefefefefefe
        return brd_front

    def get_stable(self, brd: int) -> int:
        brd_l = brd
        brd_l &= (brd_l >> 1) | 0x8080808080808080
        brd_l &= (brd_l >> 2) | 0xc0c0c0c0c0c0c0c0
        brd_l &= (brd_l >> 4) | 0xf0f0f0f0f0f0f0f0

        brd_r = brd
        brd_r &= ((brd_r << 1) % mod_64) | 0x0101010101010101
        brd_r &= ((brd_r << 2) % mod_64) | 0x0303030303030303
        brd_r &= ((brd_r << 4) % mod_64) | 0x0f0f0f0f0f0f0f0f

        brd_u = brd
        brd_u &= (brd_u >> 8) | 0xff00000000000000
        brd_u &= (brd_u >> 16) | 0xffff000000000000
        brd_u &= (brd_u >> 32) | 0xffffffff00000000

        brd_d = brd
        brd_d &= ((brd_d << 8) % mod_64) | 0x00000000000000ff
        brd_d &= ((brd_d << 16) % mod_64) | 0x000000000000ffff
        brd_d &= ((brd_d << 32) % mod_64) | 0x00000000ffffffff

        brd_ul = brd
        brd_ul &= (brd_ul >> 9) | 0xff80808080808080
        brd_ul &= (brd_ul >> 18) | 0xffffc0c0c0c0c0c0
        brd_ul &= (brd_ul >> 36) | 0xfffffffff0f0f0f0

        brd_dl = brd
        brd_dl &= ((brd_dl << 7) % mod_64) | 0x80808080808080ff
        brd_dl &= ((brd_dl << 14) % mod_64) | 0xc0c0c0c0c0c0ffff
        brd_dl &= ((brd_dl << 28) % mod_64) | 0xf0f0f0f0ffffffff

        brd_ur = brd
        brd_ur &= (brd_ur >> 7) | 0xff01010101010101
        brd_ur &= (brd_ur >> 14) | 0xffff030303030303
        brd_ur &= (brd_ur >> 28) | 0xffffffff0f0f0f0f

        brd_dr = brd
        brd_dr &= ((brd_dr << 9) % mod_64) | 0x01010101010101ff
        brd_dr &= ((brd_dr << 18) % mod_64) | 0x030303030303ffff
        brd_dr &= ((brd_dr << 36) % mod_64) | 0x0f0f0f0fffffffff

        return (brd_l | brd_r) & (brd_u | brd_d) & (brd_ul | brd_dr) & (brd_ur | brd_dl)

    # from https://www.botzone.org.cn/game/ranklist/53e1db360003e29c2ba227b8?page=0 and id of writer is stdrick
    def get_move(self, opp: int, cur: int) -> int:
        opp_inner = opp & 0x7E7E7E7E7E7E7E7E
        flip = (cur >> 1) & opp_inner
        flip |= (flip >> 1) & opp_inner
        opp_adj = opp_inner & (opp_inner >> 1)
        flip |= (flip >> 2) & opp_adj
        flip |= (flip >> 2) & opp_adj
        moves = flip >> 1
        flip = (cur << 1) & opp_inner
        flip |= (flip << 1) & opp_inner
        opp_adj = opp_inner & (opp_inner << 1)
        flip |= (flip << 2) & opp_adj
        flip |= (flip << 2) & opp_adj
        moves |= flip << 1
        flip = (cur >> 8) & opp
        flip |= (flip >> 8) & opp
        opp_adj = opp & (opp >> 8)
        flip |= (flip >> 16) & opp_adj
        flip |= (flip >> 16) & opp_adj
        moves |= flip >> 8
        flip = (cur << 8) & opp
        flip |= (flip << 8) & opp
        opp_adj = opp & (opp << 8)
        flip |= (flip << 16) & opp_adj
        flip |= (flip << 16) & opp_adj
        moves |= flip << 8
        flip = (cur >> 7) & opp_inner
        flip |= (flip >> 7) & opp_inner
        opp_adj = opp_inner & (opp_inner >> 7)
        flip |= (flip >> 14) & opp_adj
        flip |= (flip >> 14) & opp_adj
        moves |= flip >> 7
        flip = (cur << 7) & opp_inner
        flip |= (flip << 7) & opp_inner
        opp_adj = opp_inner & (opp_inner << 7)
        flip |= (flip << 14) & opp_adj
        flip |= (flip << 14) & opp_adj
        moves |= flip << 7
        flip = (cur >> 9) & opp_inner
        flip |= (flip >> 9) & opp_inner
        opp_adj = opp_inner & (opp_inner >> 9)
        flip |= (flip >> 18) & opp_adj
        flip |= (flip >> 18) & opp_adj
        moves |= flip >> 9
        flip = (cur << 9) & opp_inner
        flip |= (flip << 9) & opp_inner
        opp_adj = opp_inner & (opp_inner << 9)
        flip |= (flip << 18) & opp_adj
        flip |= (flip << 18) & opp_adj
        moves |= flip << 9
        moves &= ~(cur | opp)
        return moves % mod_64

    def dir_help(self, dir, sq):
        pos = sq
        if dir == 0:
            pos = pos >> 8
        elif dir == 1:
            pos = (pos << 8) % mod_64
        elif dir == 2:
            pos = (pos & 0xfefefefefefefefe) >> 1
        elif dir == 3:
            pos = ((pos & 0x7f7f7f7f7f7f7f7f) << 1) % mod_64
        elif dir == 4:
            pos = (pos & 0xfefefefefefefefe) >> 9
        elif dir == 5:
            pos = ((pos & 0xfefefefefefefefe) << 7) % mod_64
        elif dir == 6:
            pos = ((pos & 0x7f7f7f7f7f7f7f7f) << 9) % mod_64
        elif dir == 7:
            pos = (pos & 0x7f7f7f7f7f7f7f7f) >> 7
        return pos

    def filp_help(self, dir_1: int, sq: int, opp: int, cur: int):
        if self.dir_help(dir_1, 1 << sq) & opp:
            mask = 0
            tmp = self.dir_help(dir_1, 1 << sq)
            while tmp & opp:
                mask |= tmp
                tmp = self.dir_help(dir_1, tmp)
            if tmp & cur:
                cur ^= mask
                opp ^= mask
        return cur, opp

    def count(self, brd: int) -> int:
        result = brd - ((brd >> 1) & 0x5555555555555555)
        result = (result & 0x3333333333333333) + ((result >> 2) & 0x3333333333333333)
        result = (result + (result >> 4)) & 0x0F0F0F0F0F0F0F0F
        return ((result * 0x0101010101010101) % 0x10000000000000000) >> 56

    def getFlipped(self, sq, cur, opp):
        for i in range(8):
            cur, opp = self.filp_help(i, sq, opp, cur)
        return cur, opp

    def mirro_h(self, player):
        player = ((player >> 1) & 0x5555555555555555) | ((player << 1) & 0xAAAAAAAAAAAAAAAA)
        player = ((player >> 2) & 0x3333333333333333) | ((player << 2) & 0xCCCCCCCCCCCCCCCC)
        player = ((player >> 4) & 0x0F0F0F0F0F0F0F0F) | ((player << 4) & 0xF0F0F0F0F0F0F0F0)
        return player % mod_64

    def mirro_v(self, player):
        player = ((player >> 8) & 0x00FF00FF00FF00FF) | ((player << 8) & 0xFF00FF00FF00FF00)
        player = ((player >> 16) & 0x0000FFFF0000FFFF) | ((player << 16) & 0xFFFF0000FFFF0000)
        player = ((player >> 32) & 0x00000000FFFFFFFF) | ((player << 32) & 0xFFFFFFFF00000000)
        return player % mod_64

    # from https://github.com/Vectorized/Othello-AI
    def mirro_dia(self, player):
        temp = (player ^ (player >> 7)) & 0x00aa00aa00aa00aa
        player = player ^ temp ^ (temp << 7)
        temp = (player ^ (player >> 14)) & 0x0000cccc0000cccc
        player = player ^ temp ^ (temp << 14)
        temp = (player ^ (player >> 28)) & 0x00000000f0f0f0f0
        player = player ^ temp ^ (temp << 28)
        return player % mod_64

    # from https://github.com/Vectorized/Othello-AI
    def mirro_anti(self, player):
        temp = player ^ (player << 36)
        player ^= 0xf0f0f0f00f0f0f0f & (temp ^ (player >> 36))
        temp = 0xcccc0000cccc0000 & (player ^ (player << 18))
        player ^= temp ^ (temp >> 18)
        temp = 0xaa00aa00aa00aa00 & (player ^ (player << 9))
        return (player ^ temp ^ (temp >> 9)) % mod_64

    def mirro_h_board(self, b, w):
        return (self.mirro_h(b) << 128) + self.mirro_h(w)

    def mirro_v_board(self, b, w):
        return (self.mirro_v(b) << 128) + self.mirro_v(w)

    def mirro_dia_board(self, b, w):
        return (self.mirro_dia(b) << 128) + self.mirro_dia(w)

    def mirro_anti_board(self, b, w):
        return (self.mirro_anti(b) << 128) + self.mirro_anti(w)

    def generate_move(self, b, w):
        moves = self.get_move(w, b)
        moves_list = {}
        b_list = {}
        w_list = {}
        tot = 0
        while moves:
            moves_list[tot] = self.custom_ffs(moves) - 1
            tot += 1
            moves ^= moves & (~moves + 1)
        for i in range(tot):
            b_list[i], w_list[i] = \
                self.getFlipped(moves_list[i], b ^ 1 << moves_list[i], w)
        return moves_list, b_list, w_list

    def change(self, chessboard, color):
        b, w = 0, 0
        for i in range(8):
            for j in range(8):
                b = b << 1
                w = w << 1
                if chessboard[i][j] == color:
                    b = b + 1
                elif chessboard[i][j] == -color:
                    w = w + 1
        return b % mod_64, w % mod_64

    def eval(self, cur, opp):
        state = stageTable[self.count(cur | opp) - 4]
        result = 0
        # diag4
        for i in range(4):
            result += diag4[state][form_2_to_3[((Diag_table[i][0] & cur) * Diag_table[i][1]) >> 60 & 0xf] - \
                                   form_2_to_3[((Diag_table[i][0] & opp) * Diag_table[i][1]) >> 60 & 0xf]]
        # diag5
        for i in range(4, 8):
            if i == 6:
                result += diag5[state][form_2_to_3[((cur >> 21 & Diag_table[6][0]) * Diag_table[6][1]) >> 40 & 0x1f] - \
                                       form_2_to_3[((opp >> 21 & Diag_table[6][0]) * Diag_table[6][1]) >> 40 & 0x1f]]
            else:
                result += diag5[state][form_2_to_3[((Diag_table[i][0] & cur) * Diag_table[i][1]) >> 59 & 0x1f] - \
                                       form_2_to_3[((Diag_table[i][0] & opp) * Diag_table[i][1]) >> 59 & 0x1f]]
        # diag6
        for i in range(8, 12):
            if i != 11:
                result += diag6[state][form_2_to_3[((Diag_table[i][0] & cur) * Diag_table[i][1]) >> 58 & 0x3f] - \
                                       form_2_to_3[((Diag_table[i][0] & opp) * Diag_table[i][1]) >> 58 & 0x3f]]
            else:
                result += diag6[state][form_2_to_3[((cur >> 16 & Diag_table[11][0]) * Diag_table[11][1]) >> 58 & 0x3f] - \
                                       form_2_to_3[((opp >> 16 & Diag_table[11][0]) * Diag_table[11][1]) >> 58 & 0x3f]]
        # diag7
        for i in range(12, 15):
            result += diag7[state][form_2_to_3[((Diag_table[i][0] & cur) * Diag_table[i][1]) >> 57 & 0x7f] - \
                                   form_2_to_3[((Diag_table[i][0] & opp) * Diag_table[i][1]) >> 57 & 0x7f]]
        result += diag7[state][form_2_to_3[((cur >> 6 & Diag_table[15][0]) * Diag_table[15][1]) >> 56 & 0x7f] - \
                               form_2_to_3[((opp >> 6 & Diag_table[15][0]) * Diag_table[15][1]) >> 56 & 0x7f]]
        # dig8
        for i in range(16, 18):
            result += diag8[state][form_2_to_3[((Diag_table[i][0] & cur) * Diag_table[i][1]) >> 56 & 0xff] - \
                                   form_2_to_3[((Diag_table[i][0] & opp) * Diag_table[i][1]) >> 56 & 0xff]]
        # row
        result += lc1[state][form_2_to_3[(cur >> 56) & 0xff] - \
                             form_2_to_3[(opp >> 56) & 0xff]]
        result += lc2[state][form_2_to_3[(cur >> 48) & 0xff] - \
                             form_2_to_3[(opp >> 48) & 0xff]]
        result += lc3[state][form_2_to_3[(cur >> 40) & 0xff] - \
                             form_2_to_3[(opp >> 40) & 0xff]]
        result += lc4[state][form_2_to_3[(cur >> 32) & 0xff] - \
                             form_2_to_3[(opp >> 32) & 0xff]]
        result += lc4[state][form_2_to_3[(cur >> 24) & 0xff] - \
                             form_2_to_3[(opp >> 24) & 0xff]]
        result += lc3[state][form_2_to_3[(cur >> 16) & 0xff] - \
                             form_2_to_3[(opp >> 16) & 0xff]]
        result += lc2[state][form_2_to_3[(cur >> 8) & 0xff] - \
                             form_2_to_3[(opp >> 8) & 0xff]]
        result += lc1[state][form_2_to_3[cur & 0xff] - \
                             form_2_to_3[opp & 0xff]]
        # col
        result += lc1[state][form_2_to_3[((column_table[0][0] & cur) * column_table[0][1]) >> 56 & 0xff] - \
                             form_2_to_3[((column_table[0][0] & opp) * column_table[0][1]) >> 56 & 0xff]]
        result += lc2[state][form_2_to_3[((column_table[1][0] & cur) * column_table[1][1]) >> 56 & 0xff] - \
                             form_2_to_3[((column_table[1][0] & opp) * column_table[1][1]) >> 56 & 0xff]]
        result += lc3[state][form_2_to_3[((column_table[2][0] & cur) * column_table[2][1]) >> 56 & 0xff] - \
                             form_2_to_3[((column_table[2][0] & opp) * column_table[2][1]) >> 56 & 0xff]]
        result += lc4[state][form_2_to_3[((column_table[3][0] & cur) * column_table[3][1]) >> 56 & 0xff] - \
                             form_2_to_3[((column_table[3][0] & opp) * column_table[3][1]) >> 56 & 0xff]]
        result += lc4[state][form_2_to_3[((column_table[4][0] & cur) * column_table[4][1]) >> 56 & 0xff] - \
                             form_2_to_3[((column_table[4][0] & opp) * column_table[4][1]) >> 56 & 0xff]]
        result += lc3[state][form_2_to_3[((column_table[5][0] & cur) * column_table[5][1]) >> 56 & 0xff] - \
                             form_2_to_3[((column_table[5][0] & opp) * column_table[5][1]) >> 56 & 0xff]]
        result += lc2[state][form_2_to_3[((column_table[6][0] & cur) * column_table[6][1]) >> 56 & 0xff] - \
                             form_2_to_3[((column_table[6][0] & opp) * column_table[6][1]) >> 56 & 0xff]]
        result += lc1[state][form_2_to_3[((column_table[7][0] & cur) * column_table[7][1]) >> 56 & 0xff] - \
                             form_2_to_3[((column_table[7][0] & opp) * column_table[7][1]) >> 56 & 0xff]]
        result += parity[state][(60 - state) & 1]
        return result

    def unique_board(self, cur, opp):
        index = 7
        temp = (cur << 128) + opp
        rev_list = [i for i in range(7)]
        rev_list[0] = self.mirro_h_board(cur, opp)
        rev_list[1] = self.mirro_v_board(cur, opp)
        rev_list[2] = self.mirro_dia_board(cur, opp)
        rev_list[3] = self.mirro_anti_board(cur, opp)
        rev_list[4] = self.mirro_h_board(self.mirro_v(cur), self.mirro_v(opp))
        rev_list[5] = self.mirro_v_board(self.mirro_dia(cur), self.mirro_dia(opp))
        rev_list[6] = self.mirro_v_board(self.mirro_anti(cur), self.mirro_anti(opp))
        for i in range(len(rev_list)):
            if rev_list[i] < temp:
                temp = rev_list[i]
                index = i
        return temp, index

    def get_inver(self, b, index):
        if index == 7:
            return b
        elif index == 0:
            return self.mirro_h(b)
        elif index == 1:
            return self.mirro_v(b)
        elif index == 2:
            return self.mirro_dia(b)
        elif index == 3:
            return self.mirro_anti(b)
        elif index == 4:
            return self.mirro_v(self.mirro_h(b))
        elif index == 5:
            return self.mirro_dia(self.mirro_v(b))
        elif index == 6:
            return self.mirro_anti(self.mirro_v(b))

    def go(self, chessboard):
        self.start = time.time()
        self.stop = self.start + self.time_out
        self.candidate_list.clear()
        b, w = self.change(chessboard, self.color)
        moves_list, b_list, w_list = self.generate_move(b, w)
        for i in range(len(moves_list)):
            a = 63 - moves_list[i]
            self.candidate_list.append((int(a / 8), a % 8))
        if len(self.candidate_list) > 1:
            only_board, index = self.unique_board(b, w)
            self.trans_table2 = {}
            if only_board in openingBook:
                move = self.get_inver(openingBook[only_board][0], index)
                move = self.custom_ffs(move) - 1
                move = 63 - move
                self.candidate_list.append((int(move / 8), move % 8))
            else:
                stage = self.count(b | w)
                if stage >= 50:
                    depth = 5
                elif stage >= 54:
                    depth = 20
                else:
                    depth = 4
                for i in range(depth, 65):
                    self.trans_table = {}

                    val, move = self.alphabeta(b, w, moves_list, b_list, w_list,
                                               i, -BND, BND, 0)
                    c = 63 - move
                    if time.time() > self.stop:
                        break
                    self.candidate_list.append((int(c / 8), c % 8))
                    if val == 1 << 50:
                        break

    def alphabeta(self, cur: int, opp: int, move_list, cur_list, opp_list,
                  depth: int, alpha: int, beta: int, root=1):
        if self.count(cur | opp) == 64:
            if self.count(cur) - self.count(opp) > 0:
                return 1 << 50, -1
            else:
                return -(1 << 50), -1
        if time.time() > self.stop:
            return -1, -1
        if depth == 0:
            val = self.eval(cur, opp)
            return val, -1
        if root:
            try:
                node = self.trans_table[(cur << 64) | opp]
                if node[1] == depth:
                    pre_val = node[0]
                    type = node[2]
                    if type == 0:
                        return pre_val, node[3]
                    elif type == 1:
                        alpha = max(alpha, pre_val)
                    elif type == 2:
                        beta = min(beta, pre_val)
                    if alpha >= beta:
                        return pre_val, node[3]
            except:
                pass
        if len(move_list) == 0:
            move_next = self.get_move(cur, opp)
            moves_list_2, b_list_2, w_list_2 = self.generate_move(opp, cur)
            if self.count(move_next) > 0:
                val, move = self.alphabeta(opp, cur, moves_list_2, b_list_2, w_list_2,
                                           depth - 1, - beta, -alpha)
                return -val, move
            else:
                val = 1 << 50 if self.count(cur) - self.count(opp) > 0 else -(1 << 50)
                self.trans_table[(cur << 64) | opp] = \
                    (val, depth, 2 if val <= alpha else (1 if val >= beta else 0))
                return val, -1
        if depth >= 4:
            Val_list = []
            if cur << 64 | opp in self.trans_table2:
                ind = self.trans_table2[cur << 64 | opp]
            else:
                for i in range(len(move_list)):
                    value = self.evaluation(opp_list[i], cur_list[i])
                    Val_list.append(value)
                ind = np.argsort(Val_list)
                self.trans_table2[cur << 64 | opp] = ind
            cur_list = [cur_list[i] for i in ind[0:len(move_list)]]
            opp_list = [opp_list[i] for i in ind[0:len(move_list)]]
            move_list = [move_list[i] for i in ind[0:len(move_list)]]
        bestmove2 = move_list[0]

        b = beta
        a = alpha
        for i in range(len(move_list)):
            moves_list_2, b_list_2, w_list_2 = \
                self.generate_move(opp_list[i], cur_list[i])
            val, move_2 = self.alphabeta(opp_list[i], cur_list[i], moves_list_2, b_list_2, w_list_2, depth - 1,
                                         -b, -alpha)
            val = -val
            if i != 0 and a < val < beta:
                val, move_2 = self.alphabeta(opp_list[i], cur_list[i], moves_list_2, b_list_2, w_list_2, depth - 1,
                                             -beta, -alpha)
                val = -val
            if val > alpha:
                bestmove2 = move_list[i]
                alpha = val
            if alpha >= beta:
                break
            b = alpha + 1

        self.trans_table[(cur << 64) | opp] = (alpha, depth, 2 if alpha <= a else (1 if alpha >= beta else 0))
        return alpha, bestmove2


def place(board, x, y, color):
    if x < 0:
        return False
    board[x][y] = color
    valid = False
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
            while True:
                i -= DIR[d][0]
                j -= DIR[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid


def initBoard():
    fullInput = json.loads(input())
    requests = fullInput["requests"]
    responses = fullInput["responses"]
    board = np.zeros((8, 8), dtype=int)
    board[3][4] = board[4][3] = 1  # 白
    board[3][3] = board[4][4] = -1  # 黑
    myColor = 1
    if requests[0]["x"] >= 0:
        myColor = -1
        place(board, requests[0]["x"], requests[0]["y"], -myColor)
    turn = len(responses)
    for i in range(turn):
        place(board, responses[i]["x"], responses[i]["y"], myColor)
        place(board, requests[i + 1]["x"], requests[i + 1]["y"], -myColor)
    return board, myColor


board, myColor = initBoard()
a = AI(8, myColor, 6)
a.go(board)
if len(a.candidate_list) == 0:
    print(json.dumps({"response": {"x": -1, "y": -1}}))
else:
    move = a.candidate_list[-1]
    print(json.dumps({"response": {"x": move[0], "y": move[1]}}))

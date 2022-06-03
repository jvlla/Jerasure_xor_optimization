
encoder:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	f3 0f 1e fa          	endbr64 
    1004:	48 83 ec 08          	sub    $0x8,%rsp
    1008:	48 8b 05 d9 cf 00 00 	mov    0xcfd9(%rip),%rax        # dfe8 <__gmon_start__>
    100f:	48 85 c0             	test   %rax,%rax
    1012:	74 02                	je     1016 <_init+0x16>
    1014:	ff d0                	callq  *%rax
    1016:	48 83 c4 08          	add    $0x8,%rsp
    101a:	c3                   	retq   

Disassembly of section .plt:

0000000000001020 <.plt>:
    1020:	ff 35 82 ce 00 00    	pushq  0xce82(%rip)        # dea8 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	f2 ff 25 83 ce 00 00 	bnd jmpq *0xce83(%rip)        # deb0 <_GLOBAL_OFFSET_TABLE_+0x10>
    102d:	0f 1f 00             	nopl   (%rax)
    1030:	f3 0f 1e fa          	endbr64 
    1034:	68 00 00 00 00       	pushq  $0x0
    1039:	f2 e9 e1 ff ff ff    	bnd jmpq 1020 <.plt>
    103f:	90                   	nop
    1040:	f3 0f 1e fa          	endbr64 
    1044:	68 01 00 00 00       	pushq  $0x1
    1049:	f2 e9 d1 ff ff ff    	bnd jmpq 1020 <.plt>
    104f:	90                   	nop
    1050:	f3 0f 1e fa          	endbr64 
    1054:	68 02 00 00 00       	pushq  $0x2
    1059:	f2 e9 c1 ff ff ff    	bnd jmpq 1020 <.plt>
    105f:	90                   	nop
    1060:	f3 0f 1e fa          	endbr64 
    1064:	68 03 00 00 00       	pushq  $0x3
    1069:	f2 e9 b1 ff ff ff    	bnd jmpq 1020 <.plt>
    106f:	90                   	nop
    1070:	f3 0f 1e fa          	endbr64 
    1074:	68 04 00 00 00       	pushq  $0x4
    1079:	f2 e9 a1 ff ff ff    	bnd jmpq 1020 <.plt>
    107f:	90                   	nop
    1080:	f3 0f 1e fa          	endbr64 
    1084:	68 05 00 00 00       	pushq  $0x5
    1089:	f2 e9 91 ff ff ff    	bnd jmpq 1020 <.plt>
    108f:	90                   	nop
    1090:	f3 0f 1e fa          	endbr64 
    1094:	68 06 00 00 00       	pushq  $0x6
    1099:	f2 e9 81 ff ff ff    	bnd jmpq 1020 <.plt>
    109f:	90                   	nop
    10a0:	f3 0f 1e fa          	endbr64 
    10a4:	68 07 00 00 00       	pushq  $0x7
    10a9:	f2 e9 71 ff ff ff    	bnd jmpq 1020 <.plt>
    10af:	90                   	nop
    10b0:	f3 0f 1e fa          	endbr64 
    10b4:	68 08 00 00 00       	pushq  $0x8
    10b9:	f2 e9 61 ff ff ff    	bnd jmpq 1020 <.plt>
    10bf:	90                   	nop
    10c0:	f3 0f 1e fa          	endbr64 
    10c4:	68 09 00 00 00       	pushq  $0x9
    10c9:	f2 e9 51 ff ff ff    	bnd jmpq 1020 <.plt>
    10cf:	90                   	nop
    10d0:	f3 0f 1e fa          	endbr64 
    10d4:	68 0a 00 00 00       	pushq  $0xa
    10d9:	f2 e9 41 ff ff ff    	bnd jmpq 1020 <.plt>
    10df:	90                   	nop
    10e0:	f3 0f 1e fa          	endbr64 
    10e4:	68 0b 00 00 00       	pushq  $0xb
    10e9:	f2 e9 31 ff ff ff    	bnd jmpq 1020 <.plt>
    10ef:	90                   	nop
    10f0:	f3 0f 1e fa          	endbr64 
    10f4:	68 0c 00 00 00       	pushq  $0xc
    10f9:	f2 e9 21 ff ff ff    	bnd jmpq 1020 <.plt>
    10ff:	90                   	nop
    1100:	f3 0f 1e fa          	endbr64 
    1104:	68 0d 00 00 00       	pushq  $0xd
    1109:	f2 e9 11 ff ff ff    	bnd jmpq 1020 <.plt>
    110f:	90                   	nop
    1110:	f3 0f 1e fa          	endbr64 
    1114:	68 0e 00 00 00       	pushq  $0xe
    1119:	f2 e9 01 ff ff ff    	bnd jmpq 1020 <.plt>
    111f:	90                   	nop
    1120:	f3 0f 1e fa          	endbr64 
    1124:	68 0f 00 00 00       	pushq  $0xf
    1129:	f2 e9 f1 fe ff ff    	bnd jmpq 1020 <.plt>
    112f:	90                   	nop
    1130:	f3 0f 1e fa          	endbr64 
    1134:	68 10 00 00 00       	pushq  $0x10
    1139:	f2 e9 e1 fe ff ff    	bnd jmpq 1020 <.plt>
    113f:	90                   	nop
    1140:	f3 0f 1e fa          	endbr64 
    1144:	68 11 00 00 00       	pushq  $0x11
    1149:	f2 e9 d1 fe ff ff    	bnd jmpq 1020 <.plt>
    114f:	90                   	nop
    1150:	f3 0f 1e fa          	endbr64 
    1154:	68 12 00 00 00       	pushq  $0x12
    1159:	f2 e9 c1 fe ff ff    	bnd jmpq 1020 <.plt>
    115f:	90                   	nop
    1160:	f3 0f 1e fa          	endbr64 
    1164:	68 13 00 00 00       	pushq  $0x13
    1169:	f2 e9 b1 fe ff ff    	bnd jmpq 1020 <.plt>
    116f:	90                   	nop
    1170:	f3 0f 1e fa          	endbr64 
    1174:	68 14 00 00 00       	pushq  $0x14
    1179:	f2 e9 a1 fe ff ff    	bnd jmpq 1020 <.plt>
    117f:	90                   	nop
    1180:	f3 0f 1e fa          	endbr64 
    1184:	68 15 00 00 00       	pushq  $0x15
    1189:	f2 e9 91 fe ff ff    	bnd jmpq 1020 <.plt>
    118f:	90                   	nop
    1190:	f3 0f 1e fa          	endbr64 
    1194:	68 16 00 00 00       	pushq  $0x16
    1199:	f2 e9 81 fe ff ff    	bnd jmpq 1020 <.plt>
    119f:	90                   	nop
    11a0:	f3 0f 1e fa          	endbr64 
    11a4:	68 17 00 00 00       	pushq  $0x17
    11a9:	f2 e9 71 fe ff ff    	bnd jmpq 1020 <.plt>
    11af:	90                   	nop
    11b0:	f3 0f 1e fa          	endbr64 
    11b4:	68 18 00 00 00       	pushq  $0x18
    11b9:	f2 e9 61 fe ff ff    	bnd jmpq 1020 <.plt>
    11bf:	90                   	nop
    11c0:	f3 0f 1e fa          	endbr64 
    11c4:	68 19 00 00 00       	pushq  $0x19
    11c9:	f2 e9 51 fe ff ff    	bnd jmpq 1020 <.plt>
    11cf:	90                   	nop
    11d0:	f3 0f 1e fa          	endbr64 
    11d4:	68 1a 00 00 00       	pushq  $0x1a
    11d9:	f2 e9 41 fe ff ff    	bnd jmpq 1020 <.plt>
    11df:	90                   	nop
    11e0:	f3 0f 1e fa          	endbr64 
    11e4:	68 1b 00 00 00       	pushq  $0x1b
    11e9:	f2 e9 31 fe ff ff    	bnd jmpq 1020 <.plt>
    11ef:	90                   	nop
    11f0:	f3 0f 1e fa          	endbr64 
    11f4:	68 1c 00 00 00       	pushq  $0x1c
    11f9:	f2 e9 21 fe ff ff    	bnd jmpq 1020 <.plt>
    11ff:	90                   	nop
    1200:	f3 0f 1e fa          	endbr64 
    1204:	68 1d 00 00 00       	pushq  $0x1d
    1209:	f2 e9 11 fe ff ff    	bnd jmpq 1020 <.plt>
    120f:	90                   	nop
    1210:	f3 0f 1e fa          	endbr64 
    1214:	68 1e 00 00 00       	pushq  $0x1e
    1219:	f2 e9 01 fe ff ff    	bnd jmpq 1020 <.plt>
    121f:	90                   	nop
    1220:	f3 0f 1e fa          	endbr64 
    1224:	68 1f 00 00 00       	pushq  $0x1f
    1229:	f2 e9 f1 fd ff ff    	bnd jmpq 1020 <.plt>
    122f:	90                   	nop
    1230:	f3 0f 1e fa          	endbr64 
    1234:	68 20 00 00 00       	pushq  $0x20
    1239:	f2 e9 e1 fd ff ff    	bnd jmpq 1020 <.plt>
    123f:	90                   	nop
    1240:	f3 0f 1e fa          	endbr64 
    1244:	68 21 00 00 00       	pushq  $0x21
    1249:	f2 e9 d1 fd ff ff    	bnd jmpq 1020 <.plt>
    124f:	90                   	nop
    1250:	f3 0f 1e fa          	endbr64 
    1254:	68 22 00 00 00       	pushq  $0x22
    1259:	f2 e9 c1 fd ff ff    	bnd jmpq 1020 <.plt>
    125f:	90                   	nop
    1260:	f3 0f 1e fa          	endbr64 
    1264:	68 23 00 00 00       	pushq  $0x23
    1269:	f2 e9 b1 fd ff ff    	bnd jmpq 1020 <.plt>
    126f:	90                   	nop

Disassembly of section .plt.got:

0000000000001270 <__cxa_finalize@plt>:
    1270:	f3 0f 1e fa          	endbr64 
    1274:	f2 ff 25 7d cd 00 00 	bnd jmpq *0xcd7d(%rip)        # dff8 <__cxa_finalize@GLIBC_2.2.5>
    127b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .plt.sec:

0000000000001280 <free@plt>:
    1280:	f3 0f 1e fa          	endbr64 
    1284:	f2 ff 25 2d cc 00 00 	bnd jmpq *0xcc2d(%rip)        # deb8 <free@GLIBC_2.2.5>
    128b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001290 <putchar@plt>:
    1290:	f3 0f 1e fa          	endbr64 
    1294:	f2 ff 25 25 cc 00 00 	bnd jmpq *0xcc25(%rip)        # dec0 <putchar@GLIBC_2.2.5>
    129b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012a0 <__errno_location@plt>:
    12a0:	f3 0f 1e fa          	endbr64 
    12a4:	f2 ff 25 1d cc 00 00 	bnd jmpq *0xcc1d(%rip)        # dec8 <__errno_location@GLIBC_2.2.5>
    12ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012b0 <strcpy@plt>:
    12b0:	f3 0f 1e fa          	endbr64 
    12b4:	f2 ff 25 15 cc 00 00 	bnd jmpq *0xcc15(%rip)        # ded0 <strcpy@GLIBC_2.2.5>
    12bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012c0 <mkdir@plt>:
    12c0:	f3 0f 1e fa          	endbr64 
    12c4:	f2 ff 25 0d cc 00 00 	bnd jmpq *0xcc0d(%rip)        # ded8 <mkdir@GLIBC_2.2.5>
    12cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012d0 <fread@plt>:
    12d0:	f3 0f 1e fa          	endbr64 
    12d4:	f2 ff 25 05 cc 00 00 	bnd jmpq *0xcc05(%rip)        # dee0 <fread@GLIBC_2.2.5>
    12db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012e0 <clock_gettime@plt>:
    12e0:	f3 0f 1e fa          	endbr64 
    12e4:	f2 ff 25 fd cb 00 00 	bnd jmpq *0xcbfd(%rip)        # dee8 <clock_gettime@GLIBC_2.17>
    12eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012f0 <fclose@plt>:
    12f0:	f3 0f 1e fa          	endbr64 
    12f4:	f2 ff 25 f5 cb 00 00 	bnd jmpq *0xcbf5(%rip)        # def0 <fclose@GLIBC_2.2.5>
    12fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001300 <ctime@plt>:
    1300:	f3 0f 1e fa          	endbr64 
    1304:	f2 ff 25 ed cb 00 00 	bnd jmpq *0xcbed(%rip)        # def8 <ctime@GLIBC_2.2.5>
    130b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001310 <__stack_chk_fail@plt>:
    1310:	f3 0f 1e fa          	endbr64 
    1314:	f2 ff 25 e5 cb 00 00 	bnd jmpq *0xcbe5(%rip)        # df00 <__stack_chk_fail@GLIBC_2.4>
    131b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001320 <strchr@plt>:
    1320:	f3 0f 1e fa          	endbr64 
    1324:	f2 ff 25 dd cb 00 00 	bnd jmpq *0xcbdd(%rip)        # df08 <strchr@GLIBC_2.2.5>
    132b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001330 <strrchr@plt>:
    1330:	f3 0f 1e fa          	endbr64 
    1334:	f2 ff 25 d5 cb 00 00 	bnd jmpq *0xcbd5(%rip)        # df10 <strrchr@GLIBC_2.2.5>
    133b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001340 <__assert_fail@plt>:
    1340:	f3 0f 1e fa          	endbr64 
    1344:	f2 ff 25 cd cb 00 00 	bnd jmpq *0xcbcd(%rip)        # df18 <__assert_fail@GLIBC_2.2.5>
    134b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001350 <memset@plt>:
    1350:	f3 0f 1e fa          	endbr64 
    1354:	f2 ff 25 c5 cb 00 00 	bnd jmpq *0xcbc5(%rip)        # df20 <memset@GLIBC_2.2.5>
    135b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001360 <getcwd@plt>:
    1360:	f3 0f 1e fa          	endbr64 
    1364:	f2 ff 25 bd cb 00 00 	bnd jmpq *0xcbbd(%rip)        # df28 <getcwd@GLIBC_2.2.5>
    136b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001370 <mrand48@plt>:
    1370:	f3 0f 1e fa          	endbr64 
    1374:	f2 ff 25 b5 cb 00 00 	bnd jmpq *0xcbb5(%rip)        # df30 <mrand48@GLIBC_2.2.5>
    137b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001380 <calloc@plt>:
    1380:	f3 0f 1e fa          	endbr64 
    1384:	f2 ff 25 ad cb 00 00 	bnd jmpq *0xcbad(%rip)        # df38 <calloc@GLIBC_2.2.5>
    138b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001390 <strcmp@plt>:
    1390:	f3 0f 1e fa          	endbr64 
    1394:	f2 ff 25 a5 cb 00 00 	bnd jmpq *0xcba5(%rip)        # df40 <strcmp@GLIBC_2.2.5>
    139b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000013a0 <signal@plt>:
    13a0:	f3 0f 1e fa          	endbr64 
    13a4:	f2 ff 25 9d cb 00 00 	bnd jmpq *0xcb9d(%rip)        # df48 <signal@GLIBC_2.2.5>
    13ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000013b0 <__memcpy_chk@plt>:
    13b0:	f3 0f 1e fa          	endbr64 
    13b4:	f2 ff 25 95 cb 00 00 	bnd jmpq *0xcb95(%rip)        # df50 <__memcpy_chk@GLIBC_2.3.4>
    13bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000013c0 <memcpy@plt>:
    13c0:	f3 0f 1e fa          	endbr64 
    13c4:	f2 ff 25 8d cb 00 00 	bnd jmpq *0xcb8d(%rip)        # df58 <memcpy@GLIBC_2.14>
    13cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000013d0 <time@plt>:
    13d0:	f3 0f 1e fa          	endbr64 
    13d4:	f2 ff 25 85 cb 00 00 	bnd jmpq *0xcb85(%rip)        # df60 <time@GLIBC_2.2.5>
    13db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000013e0 <__stpcpy_chk@plt>:
    13e0:	f3 0f 1e fa          	endbr64 
    13e4:	f2 ff 25 7d cb 00 00 	bnd jmpq *0xcb7d(%rip)        # df68 <__stpcpy_chk@GLIBC_2.3.4>
    13eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000013f0 <__xstat@plt>:
    13f0:	f3 0f 1e fa          	endbr64 
    13f4:	f2 ff 25 75 cb 00 00 	bnd jmpq *0xcb75(%rip)        # df70 <__xstat@GLIBC_2.2.5>
    13fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001400 <malloc@plt>:
    1400:	f3 0f 1e fa          	endbr64 
    1404:	f2 ff 25 6d cb 00 00 	bnd jmpq *0xcb6d(%rip)        # df78 <malloc@GLIBC_2.2.5>
    140b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001410 <__isoc99_sscanf@plt>:
    1410:	f3 0f 1e fa          	endbr64 
    1414:	f2 ff 25 65 cb 00 00 	bnd jmpq *0xcb65(%rip)        # df80 <__isoc99_sscanf@GLIBC_2.7>
    141b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001420 <srand48@plt>:
    1420:	f3 0f 1e fa          	endbr64 
    1424:	f2 ff 25 5d cb 00 00 	bnd jmpq *0xcb5d(%rip)        # df88 <srand48@GLIBC_2.2.5>
    142b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001430 <__printf_chk@plt>:
    1430:	f3 0f 1e fa          	endbr64 
    1434:	f2 ff 25 55 cb 00 00 	bnd jmpq *0xcb55(%rip)        # df90 <__printf_chk@GLIBC_2.3.4>
    143b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001440 <fopen@plt>:
    1440:	f3 0f 1e fa          	endbr64 
    1444:	f2 ff 25 4d cb 00 00 	bnd jmpq *0xcb4d(%rip)        # df98 <fopen@GLIBC_2.2.5>
    144b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001450 <perror@plt>:
    1450:	f3 0f 1e fa          	endbr64 
    1454:	f2 ff 25 45 cb 00 00 	bnd jmpq *0xcb45(%rip)        # dfa0 <perror@GLIBC_2.2.5>
    145b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001460 <exit@plt>:
    1460:	f3 0f 1e fa          	endbr64 
    1464:	f2 ff 25 3d cb 00 00 	bnd jmpq *0xcb3d(%rip)        # dfa8 <exit@GLIBC_2.2.5>
    146b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001470 <fwrite@plt>:
    1470:	f3 0f 1e fa          	endbr64 
    1474:	f2 ff 25 35 cb 00 00 	bnd jmpq *0xcb35(%rip)        # dfb0 <fwrite@GLIBC_2.2.5>
    147b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001480 <__fprintf_chk@plt>:
    1480:	f3 0f 1e fa          	endbr64 
    1484:	f2 ff 25 2d cb 00 00 	bnd jmpq *0xcb2d(%rip)        # dfb8 <__fprintf_chk@GLIBC_2.3.4>
    148b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001490 <posix_memalign@plt>:
    1490:	f3 0f 1e fa          	endbr64 
    1494:	f2 ff 25 25 cb 00 00 	bnd jmpq *0xcb25(%rip)        # dfc0 <posix_memalign@GLIBC_2.2.5>
    149b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000014a0 <strdup@plt>:
    14a0:	f3 0f 1e fa          	endbr64 
    14a4:	f2 ff 25 1d cb 00 00 	bnd jmpq *0xcb1d(%rip)        # dfc8 <strdup@GLIBC_2.2.5>
    14ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000014b0 <__sprintf_chk@plt>:
    14b0:	f3 0f 1e fa          	endbr64 
    14b4:	f2 ff 25 15 cb 00 00 	bnd jmpq *0xcb15(%rip)        # dfd0 <__sprintf_chk@GLIBC_2.3.4>
    14bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .text:

00000000000014c0 <main>:
    14c0:	f3 0f 1e fa          	endbr64 
    14c4:	41 57                	push   %r15
    14c6:	41 56                	push   %r14
    14c8:	41 55                	push   %r13
    14ca:	41 54                	push   %r12
    14cc:	41 89 fc             	mov    %edi,%r12d
    14cf:	bf 03 00 00 00       	mov    $0x3,%edi
    14d4:	55                   	push   %rbp
    14d5:	53                   	push   %rbx
    14d6:	48 89 f3             	mov    %rsi,%rbx
    14d9:	48 8d 35 90 13 00 00 	lea    0x1390(%rip),%rsi        # 2870 <ctrl_bs_handler>
    14e0:	48 81 ec a8 01 00 00 	sub    $0x1a8,%rsp
    14e7:	48 8d ac 24 c0 00 00 	lea    0xc0(%rsp),%rbp
    14ee:	00 
    14ef:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    14f6:	00 00 
    14f8:	48 89 84 24 98 01 00 	mov    %rax,0x198(%rsp)
    14ff:	00 
    1500:	31 c0                	xor    %eax,%eax
    1502:	e8 99 fe ff ff       	callq  13a0 <signal@plt>
    1507:	31 ff                	xor    %edi,%edi
    1509:	48 89 ee             	mov    %rbp,%rsi
    150c:	e8 cf fd ff ff       	callq  12e0 <clock_gettime@plt>
    1511:	41 83 fc 08          	cmp    $0x8,%r12d
    1515:	0f 84 98 00 00 00    	je     15b3 <main+0xf3>

# ifdef __va_arg_pack
__fortify_function int
fprintf (FILE *__restrict __stream, const char *__restrict __fmt, ...)
{
  return __fprintf_chk (__stream, __USE_FORTIFY_LEVEL - 1, __fmt,
    151b:	48 8b 0d 1e fc 00 00 	mov    0xfc1e(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    1522:	ba 3e 00 00 00       	mov    $0x3e,%edx
    1527:	be 01 00 00 00       	mov    $0x1,%esi
    152c:	48 8d 3d 6d 8d 00 00 	lea    0x8d6d(%rip),%rdi        # a2a0 <_IO_stdin_used+0x2a0>
    1533:	e8 38 ff ff ff       	callq  1470 <fwrite@plt>
    1538:	48 8b 0d 01 fc 00 00 	mov    0xfc01(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    153f:	ba 91 00 00 00       	mov    $0x91,%edx
    1544:	be 01 00 00 00       	mov    $0x1,%esi
    1549:	48 8d 3d 90 8d 00 00 	lea    0x8d90(%rip),%rdi        # a2e0 <_IO_stdin_used+0x2e0>
    1550:	e8 1b ff ff ff       	callq  1470 <fwrite@plt>
    1555:	48 8b 0d e4 fb 00 00 	mov    0xfbe4(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    155c:	ba 2a 00 00 00       	mov    $0x2a,%edx
    1561:	be 01 00 00 00       	mov    $0x1,%esi
    1566:	48 8d 3d 0b 8e 00 00 	lea    0x8e0b(%rip),%rdi        # a378 <_IO_stdin_used+0x378>
    156d:	e8 fe fe ff ff       	callq  1470 <fwrite@plt>
    1572:	48 8b 0d c7 fb 00 00 	mov    0xfbc7(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    1579:	ba 3f 00 00 00       	mov    $0x3f,%edx
    157e:	be 01 00 00 00       	mov    $0x1,%esi
    1583:	48 8d 3d 1e 8e 00 00 	lea    0x8e1e(%rip),%rdi        # a3a8 <_IO_stdin_used+0x3a8>
    158a:	e8 e1 fe ff ff       	callq  1470 <fwrite@plt>
    158f:	48 8b 0d aa fb 00 00 	mov    0xfbaa(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    1596:	48 8d 3d 4b 8e 00 00 	lea    0x8e4b(%rip),%rdi        # a3e8 <_IO_stdin_used+0x3e8>
    159d:	ba 7c 00 00 00       	mov    $0x7c,%edx
    15a2:	be 01 00 00 00       	mov    $0x1,%esi
    15a7:	e8 c4 fe ff ff       	callq  1470 <fwrite@plt>
    15ac:	31 ff                	xor    %edi,%edi
    15ae:	e8 ad fe ff ff       	callq  1460 <exit@plt>
    15b3:	48 8b 7b 10          	mov    0x10(%rbx),%rdi
    15b7:	31 c0                	xor    %eax,%eax
    15b9:	48 8d 94 24 90 00 00 	lea    0x90(%rsp),%rdx
    15c0:	00 
    15c1:	48 8d 35 6f 8a 00 00 	lea    0x8a6f(%rip),%rsi        # a037 <_IO_stdin_used+0x37>
    15c8:	e8 43 fe ff ff       	callq  1410 <__isoc99_sscanf@plt>
    15cd:	85 c0                	test   %eax,%eax
    15cf:	74 0a                	je     15db <main+0x11b>
    15d1:	83 bc 24 90 00 00 00 	cmpl   $0x0,0x90(%rsp)
    15d8:	00 
    15d9:	7f 24                	jg     15ff <main+0x13f>
    15db:	48 8b 0d 5e fb 00 00 	mov    0xfb5e(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    15e2:	48 8d 3d 51 8a 00 00 	lea    0x8a51(%rip),%rdi        # a03a <_IO_stdin_used+0x3a>
    15e9:	ba 14 00 00 00       	mov    $0x14,%edx
    15ee:	be 01 00 00 00       	mov    $0x1,%esi
    15f3:	e8 78 fe ff ff       	callq  1470 <fwrite@plt>
    15f8:	31 ff                	xor    %edi,%edi
    15fa:	e8 61 fe ff ff       	callq  1460 <exit@plt>
    15ff:	48 8b 7b 18          	mov    0x18(%rbx),%rdi
    1603:	31 c0                	xor    %eax,%eax
    1605:	48 8d 94 24 94 00 00 	lea    0x94(%rsp),%rdx
    160c:	00 
    160d:	48 8d 35 23 8a 00 00 	lea    0x8a23(%rip),%rsi        # a037 <_IO_stdin_used+0x37>
    1614:	e8 f7 fd ff ff       	callq  1410 <__isoc99_sscanf@plt>
    1619:	85 c0                	test   %eax,%eax
    161b:	74 0a                	je     1627 <main+0x167>
    161d:	83 bc 24 94 00 00 00 	cmpl   $0x0,0x94(%rsp)
    1624:	00 
    1625:	79 24                	jns    164b <main+0x18b>
    1627:	48 8b 0d 12 fb 00 00 	mov    0xfb12(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    162e:	48 8d 3d 1a 8a 00 00 	lea    0x8a1a(%rip),%rdi        # a04f <_IO_stdin_used+0x4f>
    1635:	ba 14 00 00 00       	mov    $0x14,%edx
    163a:	be 01 00 00 00       	mov    $0x1,%esi
    163f:	e8 2c fe ff ff       	callq  1470 <fwrite@plt>
    1644:	31 ff                	xor    %edi,%edi
    1646:	e8 15 fe ff ff       	callq  1460 <exit@plt>
    164b:	48 8b 7b 28          	mov    0x28(%rbx),%rdi
    164f:	31 c0                	xor    %eax,%eax
    1651:	48 8d 94 24 98 00 00 	lea    0x98(%rsp),%rdx
    1658:	00 
    1659:	48 8d 35 d7 89 00 00 	lea    0x89d7(%rip),%rsi        # a037 <_IO_stdin_used+0x37>
    1660:	e8 ab fd ff ff       	callq  1410 <__isoc99_sscanf@plt>
    1665:	85 c0                	test   %eax,%eax
    1667:	74 0a                	je     1673 <main+0x1b3>
    1669:	83 bc 24 98 00 00 00 	cmpl   $0x0,0x98(%rsp)
    1670:	00 
    1671:	7f 24                	jg     1697 <main+0x1d7>
    1673:	48 8b 0d c6 fa 00 00 	mov    0xfac6(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    167a:	48 8d 3d e3 89 00 00 	lea    0x89e3(%rip),%rdi        # a064 <_IO_stdin_used+0x64>
    1681:	ba 15 00 00 00       	mov    $0x15,%edx
    1686:	be 01 00 00 00       	mov    $0x1,%esi
    168b:	e8 e0 fd ff ff       	callq  1470 <fwrite@plt>
    1690:	31 ff                	xor    %edi,%edi
    1692:	e8 c9 fd ff ff       	callq  1460 <exit@plt>
    1697:	48 8b 7b 30          	mov    0x30(%rbx),%rdi
    169b:	31 c0                	xor    %eax,%eax
    169d:	48 8d 94 24 9c 00 00 	lea    0x9c(%rsp),%rdx
    16a4:	00 
    16a5:	48 8d 35 8b 89 00 00 	lea    0x898b(%rip),%rsi        # a037 <_IO_stdin_used+0x37>
    16ac:	e8 5f fd ff ff       	callq  1410 <__isoc99_sscanf@plt>
    16b1:	85 c0                	test   %eax,%eax
    16b3:	74 0a                	je     16bf <main+0x1ff>
    16b5:	83 bc 24 9c 00 00 00 	cmpl   $0x0,0x9c(%rsp)
    16bc:	00 
    16bd:	79 24                	jns    16e3 <main+0x223>
    16bf:	48 8b 0d 7a fa 00 00 	mov    0xfa7a(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    16c6:	48 8d 3d 9b 8d 00 00 	lea    0x8d9b(%rip),%rdi        # a468 <_IO_stdin_used+0x468>
    16cd:	ba 1e 00 00 00       	mov    $0x1e,%edx
    16d2:	be 01 00 00 00       	mov    $0x1,%esi
    16d7:	e8 94 fd ff ff       	callq  1470 <fwrite@plt>
    16dc:	31 ff                	xor    %edi,%edi
    16de:	e8 7d fd ff ff       	callq  1460 <exit@plt>
    16e3:	48 8b 7b 38          	mov    0x38(%rbx),%rdi
    16e7:	31 c0                	xor    %eax,%eax
    16e9:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
    16f0:	00 
    16f1:	48 8d 35 82 89 00 00 	lea    0x8982(%rip),%rsi        # a07a <_IO_stdin_used+0x7a>
    16f8:	e8 13 fd ff ff       	callq  1410 <__isoc99_sscanf@plt>
    16fd:	85 c0                	test   %eax,%eax
    16ff:	0f 84 aa 01 00 00    	je     18af <main+0x3ef>
    1705:	48 8b 8c 24 a8 00 00 	mov    0xa8(%rsp),%rcx
    170c:	00 
    170d:	48 85 c9             	test   %rcx,%rcx
    1710:	0f 88 99 01 00 00    	js     18af <main+0x3ef>
    1716:	74 64                	je     177c <main+0x2bc>
    1718:	48 63 b4 24 98 00 00 	movslq 0x98(%rsp),%rsi
    171f:	00 
    1720:	48 63 94 24 90 00 00 	movslq 0x90(%rsp),%rdx
    1727:	00 
    1728:	48 63 84 24 9c 00 00 	movslq 0x9c(%rsp),%rax
    172f:	00 
    1730:	48 0f af f2          	imul   %rdx,%rsi
    1734:	85 c0                	test   %eax,%eax
    1736:	0f 85 55 05 00 00    	jne    1c91 <main+0x7d1>
    173c:	48 c1 e6 03          	shl    $0x3,%rsi
    1740:	48 89 c8             	mov    %rcx,%rax
    1743:	31 d2                	xor    %edx,%edx
    1745:	48 f7 f6             	div    %rsi
    1748:	48 85 d2             	test   %rdx,%rdx
    174b:	74 2f                	je     177c <main+0x2bc>
    174d:	48 89 cf             	mov    %rcx,%rdi
    1750:	48 89 f8             	mov    %rdi,%rax
    1753:	31 d2                	xor    %edx,%edx
    1755:	48 f7 f6             	div    %rsi
    1758:	48 85 d2             	test   %rdx,%rdx
    175b:	0f 84 b9 0f 00 00    	je     271a <main+0x125a>
    1761:	48 ff c1             	inc    %rcx
    1764:	48 89 c8             	mov    %rcx,%rax
    1767:	31 d2                	xor    %edx,%edx
    1769:	48 f7 f6             	div    %rsi
    176c:	48 ff cf             	dec    %rdi
    176f:	48 85 d2             	test   %rdx,%rdx
    1772:	75 dc                	jne    1750 <main+0x290>
    1774:	48 89 8c 24 a8 00 00 	mov    %rcx,0xa8(%rsp)
    177b:	00 
    177c:	4c 8b 63 20          	mov    0x20(%rbx),%r12
    1780:	48 8d 35 16 89 00 00 	lea    0x8916(%rip),%rsi        # a09d <_IO_stdin_used+0x9d>
    1787:	4c 89 e7             	mov    %r12,%rdi
    178a:	e8 01 fc ff ff       	callq  1390 <strcmp@plt>
    178f:	85 c0                	test   %eax,%eax
    1791:	0f 84 85 01 00 00    	je     191c <main+0x45c>
    1797:	48 8d 35 09 89 00 00 	lea    0x8909(%rip),%rsi        # a0a7 <_IO_stdin_used+0xa7>
    179e:	4c 89 e7             	mov    %r12,%rdi
    17a1:	e8 ea fb ff ff       	callq  1390 <strcmp@plt>
    17a6:	85 c0                	test   %eax,%eax
    17a8:	0f 85 25 01 00 00    	jne    18d3 <main+0x413>
    17ae:	8b 84 24 98 00 00 00 	mov    0x98(%rsp),%eax
    17b5:	8d 50 f8             	lea    -0x8(%rax),%edx
    17b8:	83 e2 f7             	and    $0xfffffff7,%edx
    17bb:	74 09                	je     17c6 <main+0x306>
    17bd:	83 f8 20             	cmp    $0x20,%eax
    17c0:	0f 85 2f 05 00 00    	jne    1cf5 <main+0x835>
    17c6:	c7 44 24 60 00 00 00 	movl   $0x0,0x60(%rsp)
    17cd:	00 
    17ce:	8b 44 24 60          	mov    0x60(%rsp),%eax
    17d2:	bf d4 03 00 00       	mov    $0x3d4,%edi
    17d7:	89 05 3b 11 01 00    	mov    %eax,0x1113b(%rip)        # 12918 <method>
    17dd:	e8 1e fc ff ff       	callq  1400 <malloc@plt>
	return __getcwd_chk (__buf, __size, __bos (__buf));

      if (__size > __bos (__buf))
	return __getcwd_chk_warn (__buf, __size, __bos (__buf));
    }
  return __getcwd_alias (__buf, __size);
    17e2:	48 89 c7             	mov    %rax,%rdi
    17e5:	be d4 03 00 00       	mov    $0x3d4,%esi
    17ea:	49 89 c4             	mov    %rax,%r12
    17ed:	e8 6e fb ff ff       	callq  1360 <getcwd@plt>
    17f2:	49 39 c4             	cmp    %rax,%r12
    17f5:	0f 85 58 0f 00 00    	jne    2753 <main+0x1293>
    17fb:	bf e8 03 00 00       	mov    $0x3e8,%edi
    1800:	e8 fb fb ff ff       	callq  1400 <malloc@plt>
    1805:	49 89 c7             	mov    %rax,%r15
#endif

__fortify_function char *
__NTH (strcpy (char *__restrict __dest, const char *__restrict __src))
{
  return __builtin___strcpy_chk (__dest, __src, __bos (__dest));
    1808:	ba e8 03 00 00       	mov    $0x3e8,%edx
    180d:	4c 89 e6             	mov    %r12,%rsi
    1810:	48 89 c7             	mov    %rax,%rdi
    1813:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    1818:	e8 c3 fb ff ff       	callq  13e0 <__stpcpy_chk@plt>


__fortify_function char *
__NTH (strcat (char *__restrict __dest, const char *__restrict __src))
{
  return __builtin___strcat_chk (__dest, __src, __bos (__dest));
    181d:	4c 89 f9             	mov    %r15,%rcx
    1820:	48 29 c1             	sub    %rax,%rcx
  return __builtin___strcpy_chk (__dest, __src, __bos (__dest));
    1823:	48 89 c7             	mov    %rax,%rdi
  return __builtin___strcat_chk (__dest, __src, __bos (__dest));
    1826:	48 81 c1 e8 03 00 00 	add    $0x3e8,%rcx
    182d:	ba 0c 00 00 00       	mov    $0xc,%edx
    1832:	48 8d 35 4a 89 00 00 	lea    0x894a(%rip),%rsi        # a183 <_IO_stdin_used+0x183>
    1839:	e8 72 fb ff ff       	callq  13b0 <__memcpy_chk@plt>
    183e:	48 8b 7b 08          	mov    0x8(%rbx),%rdi
    1842:	80 3f 2d             	cmpb   $0x2d,(%rdi)
    1845:	0f 84 fa 03 00 00    	je     1c45 <main+0x785>
    184b:	48 8d 35 3d 89 00 00 	lea    0x893d(%rip),%rsi        # a18f <_IO_stdin_used+0x18f>
    1852:	e8 e9 fb ff ff       	callq  1440 <fopen@plt>
    1857:	48 89 04 24          	mov    %rax,(%rsp)
    185b:	48 85 c0             	test   %rax,%rax
    185e:	0f 84 5b 04 00 00    	je     1cbf <main+0x7ff>
    1864:	be c0 01 00 00       	mov    $0x1c0,%esi
    1869:	48 8d 3d 38 89 00 00 	lea    0x8938(%rip),%rdi        # a1a8 <_IO_stdin_used+0x1a8>
    1870:	e8 4b fa ff ff       	callq  12c0 <mkdir@plt>
    1875:	ff c0                	inc    %eax
    1877:	0f 85 ac 00 00 00    	jne    1929 <main+0x469>
    187d:	e8 1e fa ff ff       	callq  12a0 <__errno_location@plt>
    1882:	83 38 11             	cmpl   $0x11,(%rax)
    1885:	0f 84 9e 00 00 00    	je     1929 <main+0x469>
    188b:	48 8b 0d ae f8 00 00 	mov    0xf8ae(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    1892:	48 8d 3d 7f 8d 00 00 	lea    0x8d7f(%rip),%rdi        # a618 <_IO_stdin_used+0x618>
    1899:	ba 23 00 00 00       	mov    $0x23,%edx
    189e:	be 01 00 00 00       	mov    $0x1,%esi
    18a3:	e8 c8 fb ff ff       	callq  1470 <fwrite@plt>
    18a8:	31 ff                	xor    %edi,%edi
    18aa:	e8 b1 fb ff ff       	callq  1460 <exit@plt>
    18af:	48 8b 0d 8a f8 00 00 	mov    0xf88a(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    18b6:	48 8d 3d c2 87 00 00 	lea    0x87c2(%rip),%rdi        # a07f <_IO_stdin_used+0x7f>
    18bd:	ba 1d 00 00 00       	mov    $0x1d,%edx
    18c2:	be 01 00 00 00       	mov    $0x1,%esi
    18c7:	e8 a4 fb ff ff       	callq  1470 <fwrite@plt>
    18cc:	31 ff                	xor    %edi,%edi
    18ce:	e8 8d fb ff ff       	callq  1460 <exit@plt>
    18d3:	48 8d 35 f8 87 00 00 	lea    0x87f8(%rip),%rsi        # a0d2 <_IO_stdin_used+0xd2>
    18da:	4c 89 e7             	mov    %r12,%rdi
    18dd:	e8 ae fa ff ff       	callq  1390 <strcmp@plt>
    18e2:	85 c0                	test   %eax,%eax
    18e4:	0f 85 02 0b 00 00    	jne    23ec <main+0xf2c>
    18ea:	83 bc 24 94 00 00 00 	cmpl   $0x2,0x94(%rsp)
    18f1:	02 
    18f2:	0f 84 cf 0a 00 00    	je     23c7 <main+0xf07>
    18f8:	48 8b 0d 41 f8 00 00 	mov    0xf841(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    18ff:	48 8d 3d db 87 00 00 	lea    0x87db(%rip),%rdi        # a0e1 <_IO_stdin_used+0xe1>
    1906:	ba 15 00 00 00       	mov    $0x15,%edx
    190b:	be 01 00 00 00       	mov    $0x1,%esi
    1910:	e8 5b fb ff ff       	callq  1470 <fwrite@plt>
    1915:	31 ff                	xor    %edi,%edi
    1917:	e8 44 fb ff ff       	callq  1460 <exit@plt>
    191c:	c7 44 24 60 09 00 00 	movl   $0x9,0x60(%rsp)
    1923:	00 
    1924:	e9 a5 fe ff ff       	jmpq   17ce <main+0x30e>
/* Inlined versions of the real stat and mknod functions.  */

__extern_inline int
__NTH (stat (const char *__path, struct stat *__statbuf))
{
  return __xstat (_STAT_VER, __path, __statbuf);
    1929:	48 8b 73 08          	mov    0x8(%rbx),%rsi
    192d:	48 8d 94 24 00 01 00 	lea    0x100(%rsp),%rdx
    1934:	00 
    1935:	bf 01 00 00 00       	mov    $0x1,%edi
    193a:	e8 b1 fa ff ff       	callq  13f0 <__xstat@plt>
    193f:	48 8b 84 24 30 01 00 	mov    0x130(%rsp),%rax
    1946:	00 
    1947:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
    194e:	00 
    194f:	48 63 bc 24 90 00 00 	movslq 0x90(%rsp),%rdi
    1956:	00 
    1957:	8b 8c 24 98 00 00 00 	mov    0x98(%rsp),%ecx
    195e:	8b 84 24 9c 00 00 00 	mov    0x9c(%rsp),%eax
    1965:	0f af cf             	imul   %edi,%ecx
    1968:	48 8b b4 24 a0 00 00 	mov    0xa0(%rsp),%rsi
    196f:	00 
    1970:	85 c0                	test   %eax,%eax
    1972:	0f 84 9d 02 00 00    	je     1c15 <main+0x755>
    1978:	0f af c8             	imul   %eax,%ecx
    197b:	31 d2                	xor    %edx,%edx
    197d:	48 89 f0             	mov    %rsi,%rax
    1980:	48 63 c9             	movslq %ecx,%rcx
    1983:	48 c1 e1 03          	shl    $0x3,%rcx
    1987:	48 f7 f1             	div    %rcx
    198a:	49 89 f0             	mov    %rsi,%r8
    198d:	48 85 d2             	test   %rdx,%rdx
    1990:	0f 85 4b 02 00 00    	jne    1be1 <main+0x721>
    1996:	4c 8b a4 24 a8 00 00 	mov    0xa8(%rsp),%r12
    199d:	00 
    199e:	4d 85 e4             	test   %r12,%r12
    19a1:	75 55                	jne    19f8 <main+0x538>
    19a3:	4c 89 c0             	mov    %r8,%rax
    19a6:	48 99                	cqto   
    19a8:	48 f7 ff             	idiv   %rdi
    19ab:	48 89 b4 24 a8 00 00 	mov    %rsi,0xa8(%rsp)
    19b2:	00 
    19b3:	c7 05 53 0f 01 00 01 	movl   $0x1,0x10f53(%rip)        # 12910 <readins>
    19ba:	00 00 00 
    19bd:	48 8d bc 24 b8 00 00 	lea    0xb8(%rsp),%rdi
    19c4:	00 
    19c5:	4c 89 c2             	mov    %r8,%rdx
    19c8:	be 40 00 00 00       	mov    $0x40,%esi
    19cd:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    19d2:	e8 b9 fa ff ff       	callq  1490 <posix_memalign@plt>
    19d7:	85 c0                	test   %eax,%eax
    19d9:	0f 84 04 03 00 00    	je     1ce3 <main+0x823>
    19df:	48 8d 3d 62 88 00 00 	lea    0x8862(%rip),%rdi        # a248 <_IO_stdin_used+0x248>
    19e6:	e8 65 fa ff ff       	callq  1450 <perror@plt>
    19eb:	bf 01 00 00 00       	mov    $0x1,%edi
    19f0:	e8 6b fa ff ff       	callq  1460 <exit@plt>
    19f5:	49 ff c0             	inc    %r8
    19f8:	4c 89 c0             	mov    %r8,%rax
    19fb:	48 99                	cqto   
    19fd:	49 f7 fc             	idiv   %r12
    1a00:	48 85 d2             	test   %rdx,%rdx
    1a03:	75 f0                	jne    19f5 <main+0x535>
    1a05:	49 39 f4             	cmp    %rsi,%r12
    1a08:	7d 99                	jge    19a3 <main+0x4e3>
    1a0a:	48 8d bc 24 b0 00 00 	lea    0xb0(%rsp),%rdi
    1a11:	00 
    1a12:	4c 89 e2             	mov    %r12,%rdx
    1a15:	be 40 00 00 00       	mov    $0x40,%esi
    1a1a:	89 05 f0 0e 01 00    	mov    %eax,0x10ef0(%rip)        # 12910 <readins>
    1a20:	e8 6b fa ff ff       	callq  1490 <posix_memalign@plt>
    1a25:	85 c0                	test   %eax,%eax
    1a27:	75 b6                	jne    19df <main+0x51f>
    1a29:	48 8b 84 24 b0 00 00 	mov    0xb0(%rsp),%rax
    1a30:	00 
    1a31:	48 63 8c 24 90 00 00 	movslq 0x90(%rsp),%rcx
    1a38:	00 
    1a39:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    1a3e:	4c 89 e0             	mov    %r12,%rax
    1a41:	48 99                	cqto   
    1a43:	48 f7 f9             	idiv   %rcx
    1a46:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    1a4b:	4c 8b 63 08          	mov    0x8(%rbx),%r12
    1a4f:	31 c0                	xor    %eax,%eax
    1a51:	48 83 c9 ff          	or     $0xffffffffffffffff,%rcx
    1a55:	4c 89 e7             	mov    %r12,%rdi
    1a58:	f2 ae                	repnz scas %es:(%rdi),%al
    1a5a:	48 f7 d1             	not    %rcx
    1a5d:	48 8d 79 13          	lea    0x13(%rcx),%rdi
    1a61:	49 89 cd             	mov    %rcx,%r13
    1a64:	e8 97 f9 ff ff       	callq  1400 <malloc@plt>
    1a69:	be 2f 00 00 00       	mov    $0x2f,%esi
    1a6e:	4c 89 e7             	mov    %r12,%rdi
    1a71:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
    1a76:	49 89 c7             	mov    %rax,%r15
    1a79:	e8 b2 f8 ff ff       	callq  1330 <strrchr@plt>
    1a7e:	48 85 c0             	test   %rax,%rax
    1a81:	0f 84 2b 09 00 00    	je     23b2 <main+0xef2>
    1a87:	48 8d 70 01          	lea    0x1(%rax),%rsi
  return __builtin___strcpy_chk (__dest, __src, __bos (__dest));
    1a8b:	4c 89 ff             	mov    %r15,%rdi
    1a8e:	e8 1d f8 ff ff       	callq  12b0 <strcpy@plt>
    1a93:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    1a98:	be 2e 00 00 00       	mov    $0x2e,%esi
    1a9d:	e8 7e f8 ff ff       	callq  1320 <strchr@plt>
    1aa2:	49 89 c5             	mov    %rax,%r13
    1aa5:	48 85 c0             	test   %rax,%rax
    1aa8:	0f 84 ae 09 00 00    	je     245c <main+0xf9c>
    1aae:	48 89 c7             	mov    %rax,%rdi
    1ab1:	e8 ea f9 ff ff       	callq  14a0 <strdup@plt>
    1ab6:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    1abb:	41 c6 45 00 00       	movb   $0x0,0x0(%r13)
    1ac0:	49 83 c9 ff          	or     $0xffffffffffffffff,%r9
    1ac4:	45 31 f6             	xor    %r14d,%r14d
    1ac7:	4c 89 c9             	mov    %r9,%rcx
    1aca:	44 89 f0             	mov    %r14d,%eax
    1acd:	4c 89 e7             	mov    %r12,%rdi
    1ad0:	f2 ae                	repnz scas %es:(%rdi),%al
    1ad2:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
  return __builtin___sprintf_chk (__s, __USE_FORTIFY_LEVEL - 1,
    1ad7:	4c 8d a4 24 93 01 00 	lea    0x193(%rsp),%r12
    1ade:	00 
    1adf:	48 89 ca             	mov    %rcx,%rdx
    1ae2:	4c 89 c9             	mov    %r9,%rcx
    1ae5:	f2 ae                	repnz scas %es:(%rdi),%al
    1ae7:	48 f7 d2             	not    %rdx
    1aea:	48 f7 d1             	not    %rcx
    1aed:	48 8d 7c 0a 12       	lea    0x12(%rdx,%rcx,1),%rdi
    1af2:	e8 09 f9 ff ff       	callq  1400 <malloc@plt>
    1af7:	44 8b 84 24 90 00 00 	mov    0x90(%rsp),%r8d
    1afe:	00 
    1aff:	ba 05 00 00 00       	mov    $0x5,%edx
    1b04:	be 01 00 00 00       	mov    $0x1,%esi
    1b09:	4c 89 e7             	mov    %r12,%rdi
    1b0c:	48 8d 0d 24 85 00 00 	lea    0x8524(%rip),%rcx        # a037 <_IO_stdin_used+0x37>
    1b13:	49 89 c5             	mov    %rax,%r13
    1b16:	31 c0                	xor    %eax,%eax
    1b18:	e8 93 f9 ff ff       	callq  14b0 <__sprintf_chk@plt>
    1b1d:	49 83 c9 ff          	or     $0xffffffffffffffff,%r9
    1b21:	4c 89 c9             	mov    %r9,%rcx
    1b24:	4c 89 e7             	mov    %r12,%rdi
    1b27:	44 89 f0             	mov    %r14d,%eax
    1b2a:	f2 ae                	repnz scas %es:(%rdi),%al
    1b2c:	48 63 bc 24 90 00 00 	movslq 0x90(%rsp),%rdi
    1b33:	00 
    1b34:	45 31 e4             	xor    %r12d,%r12d
    1b37:	48 c1 e7 03          	shl    $0x3,%rdi
    1b3b:	48 f7 d1             	not    %rcx
    1b3e:	8d 41 ff             	lea    -0x1(%rcx),%eax
    1b41:	89 44 24 14          	mov    %eax,0x14(%rsp)
    1b45:	e8 b6 f8 ff ff       	callq  1400 <malloc@plt>
    1b4a:	48 63 bc 24 94 00 00 	movslq 0x94(%rsp),%rdi
    1b51:	00 
    1b52:	49 89 c6             	mov    %rax,%r14
    1b55:	48 c1 e7 03          	shl    $0x3,%rdi
    1b59:	e8 a2 f8 ff ff       	callq  1400 <malloc@plt>
    1b5e:	4c 63 7c 24 50       	movslq 0x50(%rsp),%r15
    1b63:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    1b68:	48 89 5c 24 18       	mov    %rbx,0x18(%rsp)
    1b6d:	48 89 c3             	mov    %rax,%rbx
    1b70:	4c 89 f8             	mov    %r15,%rax
    1b73:	49 89 ef             	mov    %rbp,%r15
    1b76:	48 89 c5             	mov    %rax,%rbp
    1b79:	eb 1f                	jmp    1b9a <main+0x6da>
    1b7b:	48 89 ea             	mov    %rbp,%rdx
    1b7e:	be 40 00 00 00       	mov    $0x40,%esi
    1b83:	48 89 df             	mov    %rbx,%rdi
    1b86:	e8 05 f9 ff ff       	callq  1490 <posix_memalign@plt>
    1b8b:	85 c0                	test   %eax,%eax
    1b8d:	0f 85 4c fe ff ff    	jne    19df <main+0x51f>
    1b93:	41 ff c4             	inc    %r12d
    1b96:	48 83 c3 08          	add    $0x8,%rbx
    1b9a:	44 39 a4 24 94 00 00 	cmp    %r12d,0x94(%rsp)
    1ba1:	00 
    1ba2:	7f d7                	jg     1b7b <main+0x6bb>
    1ba4:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
    1bab:	00 
    1bac:	48 89 c6             	mov    %rax,%rsi
    1baf:	31 ff                	xor    %edi,%edi
    1bb1:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    1bb6:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
    1bbb:	e8 20 f7 ff ff       	callq  12e0 <clock_gettime@plt>
    1bc0:	8b 44 24 60          	mov    0x60(%rsp),%eax
    1bc4:	4c 89 fd             	mov    %r15,%rbp
    1bc7:	83 f8 08             	cmp    $0x8,%eax
    1bca:	0f 87 fd 08 00 00    	ja     24cd <main+0x100d>
    1bd0:	48 8d 15 b1 8a 00 00 	lea    0x8ab1(%rip),%rdx        # a688 <_IO_stdin_used+0x688>
    1bd7:	48 63 04 82          	movslq (%rdx,%rax,4),%rax
    1bdb:	48 01 d0             	add    %rdx,%rax
    1bde:	3e ff e0             	notrack jmpq *%rax
    1be1:	49 ff c0             	inc    %r8
    1be4:	4c 89 c0             	mov    %r8,%rax
    1be7:	31 d2                	xor    %edx,%edx
    1be9:	48 f7 f1             	div    %rcx
    1bec:	48 85 d2             	test   %rdx,%rdx
    1bef:	75 f0                	jne    1be1 <main+0x721>
    1bf1:	e9 a0 fd ff ff       	jmpq   1996 <main+0x4d6>
    1bf6:	48 8d 0d dc 8a 00 00 	lea    0x8adc(%rip),%rcx        # a6d9 <__PRETTY_FUNCTION__.5741>
    1bfd:	ba df 01 00 00       	mov    $0x1df,%edx
    1c02:	48 8d 35 22 84 00 00 	lea    0x8422(%rip),%rsi        # a02b <_IO_stdin_used+0x2b>
    1c09:	48 8d 3d 25 84 00 00 	lea    0x8425(%rip),%rdi        # a035 <_IO_stdin_used+0x35>
    1c10:	e8 2b f7 ff ff       	callq  1340 <__assert_fail@plt>
    1c15:	48 63 c9             	movslq %ecx,%rcx
    1c18:	48 c1 e1 03          	shl    $0x3,%rcx
    1c1c:	48 89 f0             	mov    %rsi,%rax
    1c1f:	31 d2                	xor    %edx,%edx
    1c21:	48 f7 f1             	div    %rcx
    1c24:	49 89 f0             	mov    %rsi,%r8
    1c27:	48 85 d2             	test   %rdx,%rdx
    1c2a:	0f 84 66 fd ff ff    	je     1996 <main+0x4d6>
    1c30:	49 ff c0             	inc    %r8
    1c33:	4c 89 c0             	mov    %r8,%rax
    1c36:	31 d2                	xor    %edx,%edx
    1c38:	48 f7 f1             	div    %rcx
    1c3b:	48 85 d2             	test   %rdx,%rdx
    1c3e:	75 f0                	jne    1c30 <main+0x770>
    1c40:	e9 51 fd ff ff       	jmpq   1996 <main+0x4d6>
    1c45:	48 ff c7             	inc    %rdi
    1c48:	31 c0                	xor    %eax,%eax
    1c4a:	48 8d 94 24 a0 00 00 	lea    0xa0(%rsp),%rdx
    1c51:	00 
    1c52:	48 8d 35 21 84 00 00 	lea    0x8421(%rip),%rsi        # a07a <_IO_stdin_used+0x7a>
    1c59:	e8 b2 f7 ff ff       	callq  1410 <__isoc99_sscanf@plt>
    1c5e:	ff c8                	dec    %eax
    1c60:	0f 85 c1 0a 00 00    	jne    2727 <main+0x1267>
    1c66:	48 83 bc 24 a0 00 00 	cmpq   $0x0,0xa0(%rsp)
    1c6d:	00 00 
    1c6f:	0f 8e b2 0a 00 00    	jle    2727 <main+0x1267>
    1c75:	31 ff                	xor    %edi,%edi
    1c77:	e8 54 f7 ff ff       	callq  13d0 <time@plt>
    1c7c:	48 89 c7             	mov    %rax,%rdi
    1c7f:	e8 9c f7 ff ff       	callq  1420 <srand48@plt>
    1c84:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
    1c8b:	00 
    1c8c:	e9 be fc ff ff       	jmpq   194f <main+0x48f>
    1c91:	48 0f af f0          	imul   %rax,%rsi
    1c95:	31 d2                	xor    %edx,%edx
    1c97:	48 89 c8             	mov    %rcx,%rax
    1c9a:	48 c1 e6 03          	shl    $0x3,%rsi
    1c9e:	48 f7 f6             	div    %rsi
    1ca1:	48 85 d2             	test   %rdx,%rdx
    1ca4:	0f 84 d2 fa ff ff    	je     177c <main+0x2bc>
    1caa:	48 ff c1             	inc    %rcx
    1cad:	48 89 c8             	mov    %rcx,%rax
    1cb0:	31 d2                	xor    %edx,%edx
    1cb2:	48 f7 f6             	div    %rsi
    1cb5:	48 85 d2             	test   %rdx,%rdx
    1cb8:	75 f0                	jne    1caa <main+0x7ea>
    1cba:	e9 b5 fa ff ff       	jmpq   1774 <main+0x2b4>
  return __fprintf_chk (__stream, __USE_FORTIFY_LEVEL - 1, __fmt,
    1cbf:	48 8b 0d 7a f4 00 00 	mov    0xf47a(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    1cc6:	48 8d 3d c5 84 00 00 	lea    0x84c5(%rip),%rdi        # a192 <_IO_stdin_used+0x192>
    1ccd:	ba 15 00 00 00       	mov    $0x15,%edx
    1cd2:	be 01 00 00 00       	mov    $0x1,%esi
    1cd7:	e8 94 f7 ff ff       	callq  1470 <fwrite@plt>
    1cdc:	31 ff                	xor    %edi,%edi
    1cde:	e8 7d f7 ff ff       	callq  1460 <exit@plt>
    1ce3:	48 8b 84 24 b8 00 00 	mov    0xb8(%rsp),%rax
    1cea:	00 
    1ceb:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    1cf0:	e9 56 fd ff ff       	jmpq   1a4b <main+0x58b>
    1cf5:	48 8b 0d 44 f4 00 00 	mov    0xf444(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    1cfc:	48 8d 3d b1 83 00 00 	lea    0x83b1(%rip),%rdi        # a0b4 <_IO_stdin_used+0xb4>
    1d03:	ba 1d 00 00 00       	mov    $0x1d,%edx
    1d08:	be 01 00 00 00       	mov    $0x1,%esi
    1d0d:	e8 5e f7 ff ff       	callq  1470 <fwrite@plt>
    1d12:	31 ff                	xor    %edi,%edi
    1d14:	e8 47 f7 ff ff       	callq  1460 <exit@plt>
    1d19:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    1d20:	e8 6b 10 00 00       	callq  2d90 <liber8tion_coding_bitmatrix>
    1d25:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    1d2c:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    1d33:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    1d3a:	48 89 c1             	mov    %rax,%rcx
    1d3d:	e8 0e 46 00 00       	callq  6350 <jerasure_smart_bitmatrix_to_schedule>
    1d42:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    1d47:	48 c7 44 24 68 00 00 	movq   $0x0,0x68(%rsp)
    1d4e:	00 00 
    1d50:	48 8d 84 24 f0 00 00 	lea    0xf0(%rsp),%rax
    1d57:	00 
    1d58:	48 89 c6             	mov    %rax,%rsi
    1d5b:	31 ff                	xor    %edi,%edi
    1d5d:	49 89 c7             	mov    %rax,%r15
    1d60:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    1d65:	e8 76 f5 ff ff       	callq  12e0 <clock_gettime@plt>
    1d6a:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
    1d6f:	4c 89 fe             	mov    %r15,%rsi
    1d72:	e8 49 7d 00 00       	callq  9ac0 <timing_delta>
    1d77:	c5 fb 58 2d 61 89 00 	vaddsd 0x8961(%rip),%xmm0,%xmm5        # a6e0 <__PRETTY_FUNCTION__.5741+0x7>
    1d7e:	00 
    1d7f:	c7 05 8b 0b 01 00 01 	movl   $0x1,0x10b8b(%rip)        # 12914 <n>
    1d86:	00 00 00 
    1d89:	c7 44 24 64 00 00 00 	movl   $0x0,0x64(%rsp)
    1d90:	00 
    1d91:	48 89 9c 24 80 00 00 	mov    %rbx,0x80(%rsp)
    1d98:	00 
    1d99:	48 89 ac 24 88 00 00 	mov    %rbp,0x88(%rsp)
    1da0:	00 
    1da1:	4c 63 64 24 50       	movslq 0x50(%rsp),%r12
    1da6:	4c 8b 7c 24 70       	mov    0x70(%rsp),%r15
    1dab:	48 8b 6c 24 78       	mov    0x78(%rsp),%rbp
    1db0:	c5 fb 11 6c 24 28    	vmovsd %xmm5,0x28(%rsp)
    1db6:	8b 05 54 0b 01 00    	mov    0x10b54(%rip),%eax        # 12910 <readins>
    1dbc:	39 05 52 0b 01 00    	cmp    %eax,0x10b52(%rip)        # 12914 <n>
    1dc2:	0f 8f 4e 03 00 00    	jg     2116 <main+0xc56>
    1dc8:	48 63 44 24 64       	movslq 0x64(%rsp),%rax
    1dcd:	48 8b 94 24 a0 00 00 	mov    0xa0(%rsp),%rdx
    1dd4:	00 
    1dd5:	48 39 d0             	cmp    %rdx,%rax
    1dd8:	0f 8c ba 02 00 00    	jl     2098 <main+0xbd8>
    1dde:	0f 84 10 03 00 00    	je     20f4 <main+0xc34>
    1de4:	8b 8c 24 90 00 00 00 	mov    0x90(%rsp),%ecx
    1deb:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
    1df0:	31 c0                	xor    %eax,%eax
    1df2:	eb 0a                	jmp    1dfe <main+0x93e>
    1df4:	49 89 14 c6          	mov    %rdx,(%r14,%rax,8)
    1df8:	48 ff c0             	inc    %rax
    1dfb:	4c 01 e2             	add    %r12,%rdx
    1dfe:	39 c1                	cmp    %eax,%ecx
    1e00:	7f f2                	jg     1df4 <main+0x934>
    1e02:	48 8b 74 24 38       	mov    0x38(%rsp),%rsi
    1e07:	31 ff                	xor    %edi,%edi
    1e09:	e8 d2 f4 ff ff       	callq  12e0 <clock_gettime@plt>
    1e0e:	8b 44 24 60          	mov    0x60(%rsp),%eax
    1e12:	83 f8 08             	cmp    $0x8,%eax
    1e15:	77 48                	ja     1e5f <main+0x99f>
    1e17:	48 8d 1d 8e 88 00 00 	lea    0x888e(%rip),%rbx        # a6ac <_IO_stdin_used+0x6ac>
    1e1e:	48 63 04 83          	movslq (%rbx,%rax,4),%rax
    1e22:	48 01 d8             	add    %rbx,%rax
    1e25:	3e ff e0             	notrack jmpq *%rax
    1e28:	8b 84 24 9c 00 00 00 	mov    0x9c(%rsp),%eax
    1e2f:	4d 89 f0             	mov    %r14,%r8
    1e32:	50                   	push   %rax
    1e33:	8b 44 24 58          	mov    0x58(%rsp),%eax
    1e37:	50                   	push   %rax
    1e38:	4c 8b 4c 24 40       	mov    0x40(%rsp),%r9
    1e3d:	48 8b 4c 24 68       	mov    0x68(%rsp),%rcx
    1e42:	8b 94 24 a8 00 00 00 	mov    0xa8(%rsp),%edx
    1e49:	8b b4 24 a4 00 00 00 	mov    0xa4(%rsp),%esi
    1e50:	8b bc 24 a0 00 00 00 	mov    0xa0(%rsp),%edi
    1e57:	e8 54 42 00 00       	callq  60b0 <jerasure_schedule_encode>
    1e5c:	5f                   	pop    %rdi
    1e5d:	41 58                	pop    %r8
    1e5f:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
    1e64:	31 ff                	xor    %edi,%edi
    1e66:	e8 75 f4 ff ff       	callq  12e0 <clock_gettime@plt>
    1e6b:	bb 01 00 00 00       	mov    $0x1,%ebx
    1e70:	eb 78                	jmp    1eea <main+0xa2a>
  return __builtin___sprintf_chk (__s, __USE_FORTIFY_LEVEL - 1,
    1e72:	56                   	push   %rsi
    1e73:	48 83 ca ff          	or     $0xffffffffffffffff,%rdx
    1e77:	4d 89 f9             	mov    %r15,%r9
    1e7a:	ff 74 24 10          	pushq  0x10(%rsp)
    1e7e:	49 89 e8             	mov    %rbp,%r8
    1e81:	48 8d 0d 32 83 00 00 	lea    0x8332(%rip),%rcx        # a1ba <_IO_stdin_used+0x1ba>
    1e88:	53                   	push   %rbx
    1e89:	be 01 00 00 00       	mov    $0x1,%esi
    1e8e:	4c 89 ef             	mov    %r13,%rdi
    1e91:	8b 44 24 2c          	mov    0x2c(%rsp),%eax
    1e95:	50                   	push   %rax
    1e96:	31 c0                	xor    %eax,%eax
    1e98:	e8 13 f6 ff ff       	callq  14b0 <__sprintf_chk@plt>
    1e9d:	48 83 c4 20          	add    $0x20,%rsp
    1ea1:	83 3d 6c 0a 01 00 01 	cmpl   $0x1,0x10a6c(%rip)        # 12914 <n>
    1ea8:	0f 84 e0 00 00 00    	je     1f8e <main+0xace>
    1eae:	48 8d 35 1d 83 00 00 	lea    0x831d(%rip),%rsi        # a1d2 <_IO_stdin_used+0x1d2>
    1eb5:	4c 89 ef             	mov    %r13,%rdi
    1eb8:	e8 83 f5 ff ff       	callq  1440 <fopen@plt>
    1ebd:	49 89 c0             	mov    %rax,%r8
    1ec0:	49 8b 7c de f8       	mov    -0x8(%r14,%rbx,8),%rdi
    1ec5:	4c 89 c1             	mov    %r8,%rcx
    1ec8:	4c 89 e2             	mov    %r12,%rdx
    1ecb:	be 01 00 00 00       	mov    $0x1,%esi
    1ed0:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
    1ed5:	e8 96 f5 ff ff       	callq  1470 <fwrite@plt>
    1eda:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
    1edf:	4c 89 c7             	mov    %r8,%rdi
    1ee2:	e8 09 f4 ff ff       	callq  12f0 <fclose@plt>
    1ee7:	48 ff c3             	inc    %rbx
    1eea:	39 9c 24 90 00 00 00 	cmp    %ebx,0x90(%rsp)
    1ef1:	0f 8c ae 00 00 00    	jl     1fa5 <main+0xae5>
    1ef7:	48 83 3c 24 00       	cmpq   $0x0,(%rsp)
    1efc:	0f 85 70 ff ff ff    	jne    1e72 <main+0x9b2>
}

__fortify_function void
__NTH (bzero (void *__dest, size_t __len))
{
  (void) __builtin___memset_chk (__dest, '\0', __len, __bos0 (__dest));
    1f02:	49 8b 7c de f8       	mov    -0x8(%r14,%rbx,8),%rdi
    1f07:	4c 89 e2             	mov    %r12,%rdx
    1f0a:	31 f6                	xor    %esi,%esi
    1f0c:	e8 3f f4 ff ff       	callq  1350 <memset@plt>
}
    1f11:	eb d4                	jmp    1ee7 <main+0xa27>
    1f13:	48 8d 0d bf 87 00 00 	lea    0x87bf(%rip),%rcx        # a6d9 <__PRETTY_FUNCTION__.5741>
    1f1a:	ba 1b 02 00 00       	mov    $0x21b,%edx
    1f1f:	48 8d 35 05 81 00 00 	lea    0x8105(%rip),%rsi        # a02b <_IO_stdin_used+0x2b>
    1f26:	48 8d 3d 08 81 00 00 	lea    0x8108(%rip),%rdi        # a035 <_IO_stdin_used+0x35>
    1f2d:	e8 0e f4 ff ff       	callq  1340 <__assert_fail@plt>
    1f32:	44 8b 44 24 50       	mov    0x50(%rsp),%r8d
    1f37:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    1f3c:	8b b4 24 98 00 00 00 	mov    0x98(%rsp),%esi
    1f43:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    1f4a:	4c 89 f2             	mov    %r14,%rdx
    1f4d:	e8 de 6b 00 00       	callq  8b30 <reed_sol_r6_encode>
    1f52:	e9 08 ff ff ff       	jmpq   1e5f <main+0x99f>
    1f57:	41 51                	push   %r9
    1f59:	4d 89 f0             	mov    %r14,%r8
    1f5c:	8b 44 24 58          	mov    0x58(%rsp),%eax
    1f60:	50                   	push   %rax
    1f61:	4c 8b 4c 24 40       	mov    0x40(%rsp),%r9
    1f66:	48 8b 4c 24 78       	mov    0x78(%rsp),%rcx
    1f6b:	8b 94 24 a8 00 00 00 	mov    0xa8(%rsp),%edx
    1f72:	8b b4 24 a4 00 00 00 	mov    0xa4(%rsp),%esi
    1f79:	8b bc 24 a0 00 00 00 	mov    0xa0(%rsp),%edi
    1f80:	e8 7b 2f 00 00       	callq  4f00 <jerasure_matrix_encode>
    1f85:	41 5a                	pop    %r10
    1f87:	41 5b                	pop    %r11
    1f89:	e9 d1 fe ff ff       	jmpq   1e5f <main+0x99f>
    1f8e:	48 8d 35 3a 82 00 00 	lea    0x823a(%rip),%rsi        # a1cf <_IO_stdin_used+0x1cf>
    1f95:	4c 89 ef             	mov    %r13,%rdi
    1f98:	e8 a3 f4 ff ff       	callq  1440 <fopen@plt>
    1f9d:	49 89 c0             	mov    %rax,%r8
    1fa0:	e9 1b ff ff ff       	jmpq   1ec0 <main+0xa00>
    1fa5:	bb 01 00 00 00       	mov    $0x1,%ebx
    1faa:	4c 89 74 24 20       	mov    %r14,0x20(%rsp)
    1faf:	49 89 de             	mov    %rbx,%r14
    1fb2:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
    1fb7:	eb 75                	jmp    202e <main+0xb6e>
    1fb9:	51                   	push   %rcx
    1fba:	48 83 ca ff          	or     $0xffffffffffffffff,%rdx
    1fbe:	4d 89 f9             	mov    %r15,%r9
    1fc1:	ff 74 24 10          	pushq  0x10(%rsp)
    1fc5:	49 89 e8             	mov    %rbp,%r8
    1fc8:	48 8d 0d 06 82 00 00 	lea    0x8206(%rip),%rcx        # a1d5 <_IO_stdin_used+0x1d5>
    1fcf:	41 56                	push   %r14
    1fd1:	be 01 00 00 00       	mov    $0x1,%esi
    1fd6:	4c 89 ef             	mov    %r13,%rdi
    1fd9:	8b 44 24 2c          	mov    0x2c(%rsp),%eax
    1fdd:	50                   	push   %rax
    1fde:	31 c0                	xor    %eax,%eax
    1fe0:	e8 cb f4 ff ff       	callq  14b0 <__sprintf_chk@plt>
    1fe5:	48 83 c4 20          	add    $0x20,%rsp
    1fe9:	83 3d 24 09 01 00 01 	cmpl   $0x1,0x10924(%rip)        # 12914 <n>
    1ff0:	74 67                	je     2059 <main+0xb99>
    1ff2:	48 8d 35 d9 81 00 00 	lea    0x81d9(%rip),%rsi        # a1d2 <_IO_stdin_used+0x1d2>
    1ff9:	4c 89 ef             	mov    %r13,%rdi
    1ffc:	e8 3f f4 ff ff       	callq  1440 <fopen@plt>
    2001:	49 89 c0             	mov    %rax,%r8
    2004:	4a 8b 7c f3 f8       	mov    -0x8(%rbx,%r14,8),%rdi
    2009:	4c 89 c1             	mov    %r8,%rcx
    200c:	4c 89 e2             	mov    %r12,%rdx
    200f:	be 01 00 00 00       	mov    $0x1,%esi
    2014:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
    2019:	e8 52 f4 ff ff       	callq  1470 <fwrite@plt>
    201e:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
    2023:	4c 89 c7             	mov    %r8,%rdi
    2026:	e8 c5 f2 ff ff       	callq  12f0 <fclose@plt>
    202b:	49 ff c6             	inc    %r14
    202e:	44 39 b4 24 94 00 00 	cmp    %r14d,0x94(%rsp)
    2035:	00 
    2036:	7c 35                	jl     206d <main+0xbad>
    2038:	48 83 3c 24 00       	cmpq   $0x0,(%rsp)
    203d:	0f 85 76 ff ff ff    	jne    1fb9 <main+0xaf9>
  (void) __builtin___memset_chk (__dest, '\0', __len, __bos0 (__dest));
    2043:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    2048:	4c 89 e2             	mov    %r12,%rdx
    204b:	4a 8b 7c f0 f8       	mov    -0x8(%rax,%r14,8),%rdi
    2050:	31 f6                	xor    %esi,%esi
    2052:	e8 f9 f2 ff ff       	callq  1350 <memset@plt>
}
    2057:	eb d2                	jmp    202b <main+0xb6b>
    2059:	48 8d 35 6f 81 00 00 	lea    0x816f(%rip),%rsi        # a1cf <_IO_stdin_used+0x1cf>
    2060:	4c 89 ef             	mov    %r13,%rdi
    2063:	e8 d8 f3 ff ff       	callq  1440 <fopen@plt>
    2068:	49 89 c0             	mov    %rax,%r8
    206b:	eb 97                	jmp    2004 <main+0xb44>
    206d:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
    2072:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
    2077:	ff 05 97 08 01 00    	incl   0x10897(%rip)        # 12914 <n>
    207d:	4c 8b 74 24 20       	mov    0x20(%rsp),%r14
    2082:	e8 39 7a 00 00       	callq  9ac0 <timing_delta>
    2087:	c5 fb 58 54 24 28    	vaddsd 0x28(%rsp),%xmm0,%xmm2
    208d:	c5 fb 11 54 24 28    	vmovsd %xmm2,0x28(%rsp)
    2093:	e9 1e fd ff ff       	jmpq   1db6 <main+0x8f6>
    2098:	4c 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%r8
    209f:	00 
    20a0:	48 8b 0c 24          	mov    (%rsp),%rcx
    20a4:	4c 01 c0             	add    %r8,%rax
    20a7:	48 39 c2             	cmp    %rax,%rdx
    20aa:	be 01 00 00 00       	mov    $0x1,%esi
    20af:	4c 89 c2             	mov    %r8,%rdx
    20b2:	7d 2d                	jge    20e1 <main+0xc21>
    20b4:	48 8b 5c 24 40       	mov    0x40(%rsp),%rbx
    20b9:	48 89 df             	mov    %rbx,%rdi
    20bc:	e8 af 08 00 00       	callq  2970 <jfread>
    20c1:	48 98                	cltq   
    20c3:	48 8b 94 24 a8 00 00 	mov    0xa8(%rsp),%rdx
    20ca:	00 
    20cb:	48 89 d9             	mov    %rbx,%rcx
    20ce:	eb 07                	jmp    20d7 <main+0xc17>
    20d0:	c6 04 01 30          	movb   $0x30,(%rcx,%rax,1)
    20d4:	48 ff c0             	inc    %rax
    20d7:	48 39 c2             	cmp    %rax,%rdx
    20da:	7f f4                	jg     20d0 <main+0xc10>
    20dc:	e9 03 fd ff ff       	jmpq   1de4 <main+0x924>
    20e1:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    20e6:	e8 85 08 00 00       	callq  2970 <jfread>
    20eb:	01 44 24 64          	add    %eax,0x64(%rsp)
    20ef:	e9 f0 fc ff ff       	jmpq   1de4 <main+0x924>
    20f4:	48 8b 94 24 a8 00 00 	mov    0xa8(%rsp),%rdx
    20fb:	00 
    20fc:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    2101:	31 c0                	xor    %eax,%eax
    2103:	eb 07                	jmp    210c <main+0xc4c>
    2105:	c6 04 01 30          	movb   $0x30,(%rcx,%rax,1)
    2109:	48 ff c0             	inc    %rax
    210c:	48 39 d0             	cmp    %rdx,%rax
    210f:	7c f4                	jl     2105 <main+0xc45>
    2111:	e9 ce fc ff ff       	jmpq   1de4 <main+0x924>
    2116:	48 83 3c 24 00       	cmpq   $0x0,(%rsp)
    211b:	48 8b 9c 24 80 00 00 	mov    0x80(%rsp),%rbx
    2122:	00 
    2123:	48 8b ac 24 88 00 00 	mov    0x88(%rsp),%rbp
    212a:	00 
    212b:	0f 84 04 01 00 00    	je     2235 <main+0xd75>
    2131:	4c 8b 4c 24 70       	mov    0x70(%rsp),%r9
    2136:	4c 8b 44 24 78       	mov    0x78(%rsp),%r8
    213b:	48 8d 0d a8 80 00 00 	lea    0x80a8(%rip),%rcx        # a1ea <_IO_stdin_used+0x1ea>
    2142:	48 83 ca ff          	or     $0xffffffffffffffff,%rdx
    2146:	be 01 00 00 00       	mov    $0x1,%esi
    214b:	4c 89 ef             	mov    %r13,%rdi
    214e:	31 c0                	xor    %eax,%eax
    2150:	e8 5b f3 ff ff       	callq  14b0 <__sprintf_chk@plt>
    2155:	48 8d 35 73 80 00 00 	lea    0x8073(%rip),%rsi        # a1cf <_IO_stdin_used+0x1cf>
    215c:	4c 89 ef             	mov    %r13,%rdi
    215f:	e8 dc f2 ff ff       	callq  1440 <fopen@plt>
  return __fprintf_chk (__stream, __USE_FORTIFY_LEVEL - 1, __fmt,
    2164:	48 8b 4b 08          	mov    0x8(%rbx),%rcx
    2168:	49 89 c4             	mov    %rax,%r12
    216b:	48 89 c7             	mov    %rax,%rdi
    216e:	48 8d 15 90 7e 00 00 	lea    0x7e90(%rip),%rdx        # a005 <_IO_stdin_used+0x5>
    2175:	be 01 00 00 00       	mov    $0x1,%esi
    217a:	31 c0                	xor    %eax,%eax
    217c:	e8 ff f2 ff ff       	callq  1480 <__fprintf_chk@plt>
    2181:	48 8b 8c 24 a0 00 00 	mov    0xa0(%rsp),%rcx
    2188:	00 
    2189:	48 8d 15 7c 80 00 00 	lea    0x807c(%rip),%rdx        # a20c <_IO_stdin_used+0x20c>
    2190:	be 01 00 00 00       	mov    $0x1,%esi
    2195:	4c 89 e7             	mov    %r12,%rdi
    2198:	31 c0                	xor    %eax,%eax
    219a:	e8 e1 f2 ff ff       	callq  1480 <__fprintf_chk@plt>
    219f:	ff b4 24 a8 00 00 00 	pushq  0xa8(%rsp)
    21a6:	48 8d 15 53 80 00 00 	lea    0x8053(%rip),%rdx        # a200 <_IO_stdin_used+0x200>
    21ad:	be 01 00 00 00       	mov    $0x1,%esi
    21b2:	8b 84 24 a4 00 00 00 	mov    0xa4(%rsp),%eax
    21b9:	4c 89 e7             	mov    %r12,%rdi
    21bc:	50                   	push   %rax
    21bd:	31 c0                	xor    %eax,%eax
    21bf:	44 8b 8c 24 a8 00 00 	mov    0xa8(%rsp),%r9d
    21c6:	00 
    21c7:	44 8b 84 24 a4 00 00 	mov    0xa4(%rsp),%r8d
    21ce:	00 
    21cf:	8b 8c 24 a0 00 00 00 	mov    0xa0(%rsp),%ecx
    21d6:	e8 a5 f2 ff ff       	callq  1480 <__fprintf_chk@plt>
    21db:	48 8b 4b 20          	mov    0x20(%rbx),%rcx
    21df:	48 8d 15 1f 7e 00 00 	lea    0x7e1f(%rip),%rdx        # a005 <_IO_stdin_used+0x5>
    21e6:	be 01 00 00 00       	mov    $0x1,%esi
    21eb:	4c 89 e7             	mov    %r12,%rdi
    21ee:	31 c0                	xor    %eax,%eax
    21f0:	e8 8b f2 ff ff       	callq  1480 <__fprintf_chk@plt>
    21f5:	8b 4c 24 70          	mov    0x70(%rsp),%ecx
    21f9:	48 8d 15 1a 7e 00 00 	lea    0x7e1a(%rip),%rdx        # a01a <_IO_stdin_used+0x1a>
    2200:	be 01 00 00 00       	mov    $0x1,%esi
    2205:	4c 89 e7             	mov    %r12,%rdi
    2208:	31 c0                	xor    %eax,%eax
    220a:	e8 71 f2 ff ff       	callq  1480 <__fprintf_chk@plt>
    220f:	8b 0d fb 06 01 00    	mov    0x106fb(%rip),%ecx        # 12910 <readins>
    2215:	48 8d 15 fe 7d 00 00 	lea    0x7dfe(%rip),%rdx        # a01a <_IO_stdin_used+0x1a>
    221c:	be 01 00 00 00       	mov    $0x1,%esi
    2221:	4c 89 e7             	mov    %r12,%rdi
    2224:	31 c0                	xor    %eax,%eax
    2226:	e8 55 f2 ff ff       	callq  1480 <__fprintf_chk@plt>
    222b:	4c 89 e7             	mov    %r12,%rdi
    222e:	e8 bd f0 ff ff       	callq  12f0 <fclose@plt>
    2233:	58                   	pop    %rax
    2234:	5a                   	pop    %rdx
    2235:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    223a:	4c 8d a4 24 d0 00 00 	lea    0xd0(%rsp),%r12
    2241:	00 
    2242:	e8 39 f0 ff ff       	callq  1280 <free@plt>
    2247:	4c 89 ef             	mov    %r13,%rdi
    224a:	e8 31 f0 ff ff       	callq  1280 <free@plt>
    224f:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    2254:	e8 27 f0 ff ff       	callq  1280 <free@plt>
    2259:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
    225e:	e8 1d f0 ff ff       	callq  1280 <free@plt>
    2263:	31 ff                	xor    %edi,%edi
    2265:	4c 89 e6             	mov    %r12,%rsi
    2268:	e8 73 f0 ff ff       	callq  12e0 <clock_gettime@plt>
    226d:	4c 89 e6             	mov    %r12,%rsi
    2270:	48 89 ef             	mov    %rbp,%rdi
    2273:	e8 48 78 00 00       	callq  9ac0 <timing_delta>
    2278:	c5 d9 57 e4          	vxorpd %xmm4,%xmm4,%xmm4
    227c:	c5 fb 11 04 24       	vmovsd %xmm0,(%rsp)
    2281:	c4 e1 db 2a 84 24 a0 	vcvtsi2sdq 0xa0(%rsp),%xmm4,%xmm0
    2288:	00 00 00 
    228b:	c5 fb 10 0d 55 84 00 	vmovsd 0x8455(%rip),%xmm1        # a6e8 <__PRETTY_FUNCTION__.5741+0xf>
    2292:	00 
}

__fortify_function int
printf (const char *__restrict __fmt, ...)
{
  return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
    2293:	48 8d 35 78 7f 00 00 	lea    0x7f78(%rip),%rsi        # a212 <_IO_stdin_used+0x212>
    229a:	bf 01 00 00 00       	mov    $0x1,%edi
    229f:	c5 fb 59 c1          	vmulsd %xmm1,%xmm0,%xmm0
    22a3:	b8 01 00 00 00       	mov    $0x1,%eax
    22a8:	c5 fb 59 c1          	vmulsd %xmm1,%xmm0,%xmm0
    22ac:	c5 fb 5e 44 24 28    	vdivsd 0x28(%rsp),%xmm0,%xmm0
    22b2:	e8 79 f1 ff ff       	callq  1430 <__printf_chk@plt>
    22b7:	c5 d9 57 e4          	vxorpd %xmm4,%xmm4,%xmm4
    22bb:	c4 e1 db 2a 84 24 a0 	vcvtsi2sdq 0xa0(%rsp),%xmm4,%xmm0
    22c2:	00 00 00 
    22c5:	48 8b 05 1c 84 00 00 	mov    0x841c(%rip),%rax        # a6e8 <__PRETTY_FUNCTION__.5741+0xf>
    22cc:	48 8d 35 5a 7f 00 00 	lea    0x7f5a(%rip),%rsi        # a22d <_IO_stdin_used+0x22d>
    22d3:	c4 e1 f9 6e c8       	vmovq  %rax,%xmm1
    22d8:	c5 fb 59 c1          	vmulsd %xmm1,%xmm0,%xmm0
    22dc:	bf 01 00 00 00       	mov    $0x1,%edi
    22e1:	b8 01 00 00 00       	mov    $0x1,%eax
    22e6:	c5 fb 59 c1          	vmulsd %xmm1,%xmm0,%xmm0
    22ea:	c5 fb 5e 04 24       	vdivsd (%rsp),%xmm0,%xmm0
    22ef:	e8 3c f1 ff ff       	callq  1430 <__printf_chk@plt>
    22f4:	48 8b 84 24 98 01 00 	mov    0x198(%rsp),%rax
    22fb:	00 
    22fc:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    2303:	00 00 
    2305:	0f 85 43 04 00 00    	jne    274e <main+0x128e>
    230b:	48 81 c4 a8 01 00 00 	add    $0x1a8,%rsp
    2312:	5b                   	pop    %rbx
    2313:	5d                   	pop    %rbp
    2314:	41 5c                	pop    %r12
    2316:	41 5d                	pop    %r13
    2318:	41 5e                	pop    %r14
    231a:	31 c0                	xor    %eax,%eax
    231c:	41 5f                	pop    %r15
    231e:	c3                   	retq   
    231f:	8b b4 24 98 00 00 00 	mov    0x98(%rsp),%esi
    2326:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    232d:	e8 9e 0e 00 00       	callq  31d0 <blaum_roth_coding_bitmatrix>
    2332:	e9 ee f9 ff ff       	jmpq   1d25 <main+0x865>
    2337:	8b b4 24 98 00 00 00 	mov    0x98(%rsp),%esi
    233e:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    2345:	e8 e6 08 00 00       	callq  2c30 <liberation_coding_bitmatrix>
    234a:	e9 d6 f9 ff ff       	jmpq   1d25 <main+0x865>
    234f:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    2356:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    235d:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    2364:	e8 e7 74 00 00       	callq  9850 <cauchy_good_general_coding_matrix>
    2369:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    2370:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    2377:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    237e:	48 89 c1             	mov    %rax,%rcx
    2381:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
    2386:	e8 85 12 00 00       	callq  3610 <jerasure_matrix_to_bitmatrix>
    238b:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    2392:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    2399:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    23a0:	48 89 c1             	mov    %rax,%rcx
    23a3:	e8 a8 3f 00 00       	callq  6350 <jerasure_smart_bitmatrix_to_schedule>
    23a8:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    23ad:	e9 9e f9 ff ff       	jmpq   1d50 <main+0x890>
    23b2:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    23b7:	4c 89 ea             	mov    %r13,%rdx
    23ba:	4c 89 e6             	mov    %r12,%rsi
    23bd:	e8 fe ef ff ff       	callq  13c0 <memcpy@plt>
    23c2:	e9 cc f6 ff ff       	jmpq   1a93 <main+0x5d3>
    23c7:	8b 84 24 98 00 00 00 	mov    0x98(%rsp),%eax
    23ce:	8d 50 f8             	lea    -0x8(%rax),%edx
    23d1:	83 e2 f7             	and    $0xfffffff7,%edx
    23d4:	74 09                	je     23df <main+0xf1f>
    23d6:	83 f8 20             	cmp    $0x20,%eax
    23d9:	0f 85 16 f9 ff ff    	jne    1cf5 <main+0x835>
    23df:	c7 44 24 60 01 00 00 	movl   $0x1,0x60(%rsp)
    23e6:	00 
    23e7:	e9 e2 f3 ff ff       	jmpq   17ce <main+0x30e>
    23ec:	48 8d 35 04 7d 00 00 	lea    0x7d04(%rip),%rsi        # a0f7 <_IO_stdin_used+0xf7>
    23f3:	4c 89 e7             	mov    %r12,%rdi
    23f6:	e8 95 ef ff ff       	callq  1390 <strcmp@plt>
    23fb:	85 c0                	test   %eax,%eax
    23fd:	0f 85 9c 00 00 00    	jne    249f <main+0xfdf>
    2403:	83 bc 24 9c 00 00 00 	cmpl   $0x0,0x9c(%rsp)
    240a:	00 
    240b:	c7 44 24 60 02 00 00 	movl   $0x2,0x60(%rsp)
    2412:	00 
    2413:	0f 85 b5 f3 ff ff    	jne    17ce <main+0x30e>
  return __fprintf_chk (__stream, __USE_FORTIFY_LEVEL - 1, __fmt,
    2419:	48 8b 0d 20 ed 00 00 	mov    0xed20(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    2420:	48 8d 3d dc 7c 00 00 	lea    0x7cdc(%rip),%rdi        # a103 <_IO_stdin_used+0x103>
    2427:	ba 19 00 00 00       	mov    $0x19,%edx
    242c:	be 01 00 00 00       	mov    $0x1,%esi
    2431:	e8 3a f0 ff ff       	callq  1470 <fwrite@plt>
    2436:	31 ff                	xor    %edi,%edi
    2438:	e8 23 f0 ff ff       	callq  1460 <exit@plt>
    243d:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    2444:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    244b:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    2452:	e8 f9 6f 00 00       	callq  9450 <cauchy_original_coding_matrix>
    2457:	e9 0d ff ff ff       	jmpq   2369 <main+0xea9>
    245c:	48 8d 3d c7 7b 00 00 	lea    0x7bc7(%rip),%rdi        # a02a <_IO_stdin_used+0x2a>
    2463:	e8 38 f0 ff ff       	callq  14a0 <strdup@plt>
    2468:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    246d:	e9 4e f6 ff ff       	jmpq   1ac0 <main+0x600>
    2472:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    2479:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    2480:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    2487:	e8 94 6d 00 00       	callq  9220 <reed_sol_vandermonde_coding_matrix>
    248c:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
    2491:	48 c7 44 24 58 00 00 	movq   $0x0,0x58(%rsp)
    2498:	00 00 
    249a:	e9 b1 f8 ff ff       	jmpq   1d50 <main+0x890>
    249f:	48 8d 35 77 7c 00 00 	lea    0x7c77(%rip),%rsi        # a11d <_IO_stdin_used+0x11d>
    24a6:	4c 89 e7             	mov    %r12,%rdi
    24a9:	e8 e2 ee ff ff       	callq  1390 <strcmp@plt>
    24ae:	85 c0                	test   %eax,%eax
    24b0:	75 32                	jne    24e4 <main+0x1024>
    24b2:	83 bc 24 9c 00 00 00 	cmpl   $0x0,0x9c(%rsp)
    24b9:	00 
    24ba:	0f 84 59 ff ff ff    	je     2419 <main+0xf59>
    24c0:	c7 44 24 60 03 00 00 	movl   $0x3,0x60(%rsp)
    24c7:	00 
    24c8:	e9 01 f3 ff ff       	jmpq   17ce <main+0x30e>
    24cd:	48 c7 44 24 58 00 00 	movq   $0x0,0x58(%rsp)
    24d4:	00 00 
    24d6:	48 c7 44 24 68 00 00 	movq   $0x0,0x68(%rsp)
    24dd:	00 00 
    24df:	e9 6c f8 ff ff       	jmpq   1d50 <main+0x890>
    24e4:	48 8d 35 3e 7c 00 00 	lea    0x7c3e(%rip),%rsi        # a129 <_IO_stdin_used+0x129>
    24eb:	4c 89 e7             	mov    %r12,%rdi
    24ee:	e8 9d ee ff ff       	callq  1390 <strcmp@plt>
    24f3:	85 c0                	test   %eax,%eax
    24f5:	75 67                	jne    255e <main+0x109e>
    24f7:	8b bc 24 98 00 00 00 	mov    0x98(%rsp),%edi
    24fe:	39 bc 24 90 00 00 00 	cmp    %edi,0x90(%rsp)
    2505:	7f 33                	jg     253a <main+0x107a>
    2507:	83 ff 02             	cmp    $0x2,%edi
    250a:	7e 0a                	jle    2516 <main+0x1056>
    250c:	40 f6 c7 01          	test   $0x1,%dil
    2510:	0f 85 f6 00 00 00    	jne    260c <main+0x114c>
    2516:	48 8b 0d 23 ec 00 00 	mov    0xec23(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    251d:	48 8d 3d 8c 7f 00 00 	lea    0x7f8c(%rip),%rdi        # a4b0 <_IO_stdin_used+0x4b0>
    2524:	ba 2f 00 00 00       	mov    $0x2f,%edx
    2529:	be 01 00 00 00       	mov    $0x1,%esi
    252e:	e8 3d ef ff ff       	callq  1470 <fwrite@plt>
    2533:	31 ff                	xor    %edi,%edi
    2535:	e8 26 ef ff ff       	callq  1460 <exit@plt>
    253a:	48 8b 0d ff eb 00 00 	mov    0xebff(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    2541:	48 8d 3d 40 7f 00 00 	lea    0x7f40(%rip),%rdi        # a488 <_IO_stdin_used+0x488>
    2548:	ba 22 00 00 00       	mov    $0x22,%edx
    254d:	be 01 00 00 00       	mov    $0x1,%esi
    2552:	e8 19 ef ff ff       	callq  1470 <fwrite@plt>
    2557:	31 ff                	xor    %edi,%edi
    2559:	e8 02 ef ff ff       	callq  1460 <exit@plt>
    255e:	48 8d 35 cf 7b 00 00 	lea    0x7bcf(%rip),%rsi        # a134 <_IO_stdin_used+0x134>
    2565:	4c 89 e7             	mov    %r12,%rdi
    2568:	e8 23 ee ff ff       	callq  1390 <strcmp@plt>
    256d:	85 c0                	test   %eax,%eax
    256f:	75 44                	jne    25b5 <main+0x10f5>
    2571:	8b 84 24 98 00 00 00 	mov    0x98(%rsp),%eax
    2578:	39 84 24 90 00 00 00 	cmp    %eax,0x90(%rsp)
    257f:	7f b9                	jg     253a <main+0x107a>
    2581:	83 f8 02             	cmp    $0x2,%eax
    2584:	7e 0b                	jle    2591 <main+0x10d1>
    2586:	8d 78 01             	lea    0x1(%rax),%edi
    2589:	a8 01                	test   $0x1,%al
    258b:	0f 84 cb 00 00 00    	je     265c <main+0x119c>
    2591:	48 8b 0d a8 eb 00 00 	mov    0xeba8(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    2598:	48 8d 3d 71 7f 00 00 	lea    0x7f71(%rip),%rdi        # a510 <_IO_stdin_used+0x510>
    259f:	ba 31 00 00 00       	mov    $0x31,%edx
    25a4:	be 01 00 00 00       	mov    $0x1,%esi
    25a9:	e8 c2 ee ff ff       	callq  1470 <fwrite@plt>
    25ae:	31 ff                	xor    %edi,%edi
    25b0:	e8 ab ee ff ff       	callq  1460 <exit@plt>
    25b5:	48 8d 35 83 7b 00 00 	lea    0x7b83(%rip),%rsi        # a13f <_IO_stdin_used+0x13f>
    25bc:	4c 89 e7             	mov    %r12,%rdi
    25bf:	e8 cc ed ff ff       	callq  1390 <strcmp@plt>
    25c4:	85 c0                	test   %eax,%eax
    25c6:	0f 85 0f 01 00 00    	jne    26db <main+0x121b>
    25cc:	83 bc 24 9c 00 00 00 	cmpl   $0x0,0x9c(%rsp)
    25d3:	00 
    25d4:	0f 84 dd 00 00 00    	je     26b7 <main+0x11f7>
    25da:	83 bc 24 98 00 00 00 	cmpl   $0x8,0x98(%rsp)
    25e1:	08 
    25e2:	0f 84 a1 00 00 00    	je     2689 <main+0x11c9>
    25e8:	48 8b 0d 51 eb 00 00 	mov    0xeb51(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    25ef:	48 8d 3d 6d 7b 00 00 	lea    0x7b6d(%rip),%rdi        # a163 <_IO_stdin_used+0x163>
    25f6:	ba 0f 00 00 00       	mov    $0xf,%edx
    25fb:	be 01 00 00 00       	mov    $0x1,%esi
    2600:	e8 6b ee ff ff       	callq  1470 <fwrite@plt>
    2605:	31 ff                	xor    %edi,%edi
    2607:	e8 54 ee ff ff       	callq  1460 <exit@plt>
    260c:	e8 bf 03 00 00       	callq  29d0 <is_prime>
    2611:	85 c0                	test   %eax,%eax
    2613:	0f 84 fd fe ff ff    	je     2516 <main+0x1056>
    2619:	8b 84 24 9c 00 00 00 	mov    0x9c(%rsp),%eax
    2620:	85 c0                	test   %eax,%eax
    2622:	0f 84 f1 fd ff ff    	je     2419 <main+0xf59>
    2628:	c7 44 24 60 04 00 00 	movl   $0x4,0x60(%rsp)
    262f:	00 
    2630:	a8 07                	test   $0x7,%al
    2632:	0f 84 96 f1 ff ff    	je     17ce <main+0x30e>
    2638:	48 8b 0d 01 eb 00 00 	mov    0xeb01(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    263f:	48 8d 3d 9a 7e 00 00 	lea    0x7e9a(%rip),%rdi        # a4e0 <_IO_stdin_used+0x4e0>
    2646:	ba 2e 00 00 00       	mov    $0x2e,%edx
    264b:	be 01 00 00 00       	mov    $0x1,%esi
    2650:	e8 1b ee ff ff       	callq  1470 <fwrite@plt>
    2655:	31 ff                	xor    %edi,%edi
    2657:	e8 04 ee ff ff       	callq  1460 <exit@plt>
    265c:	e8 6f 03 00 00       	callq  29d0 <is_prime>
    2661:	85 c0                	test   %eax,%eax
    2663:	0f 84 28 ff ff ff    	je     2591 <main+0x10d1>
    2669:	8b 84 24 9c 00 00 00 	mov    0x9c(%rsp),%eax
    2670:	85 c0                	test   %eax,%eax
    2672:	0f 84 a1 fd ff ff    	je     2419 <main+0xf59>
    2678:	a8 07                	test   $0x7,%al
    267a:	75 bc                	jne    2638 <main+0x1178>
    267c:	c7 44 24 60 05 00 00 	movl   $0x5,0x60(%rsp)
    2683:	00 
    2684:	e9 45 f1 ff ff       	jmpq   17ce <main+0x30e>
    2689:	83 bc 24 94 00 00 00 	cmpl   $0x2,0x94(%rsp)
    2690:	02 
    2691:	74 6c                	je     26ff <main+0x123f>
    2693:	48 8b 0d a6 ea 00 00 	mov    0xeaa6(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    269a:	48 8d 3d d2 7a 00 00 	lea    0x7ad2(%rip),%rdi        # a173 <_IO_stdin_used+0x173>
    26a1:	ba 0f 00 00 00       	mov    $0xf,%edx
    26a6:	be 01 00 00 00       	mov    $0x1,%esi
    26ab:	e8 c0 ed ff ff       	callq  1470 <fwrite@plt>
    26b0:	31 ff                	xor    %edi,%edi
    26b2:	e8 a9 ed ff ff       	callq  1460 <exit@plt>
    26b7:	48 8b 0d 82 ea 00 00 	mov    0xea82(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    26be:	48 8d 3d 85 7a 00 00 	lea    0x7a85(%rip),%rdi        # a14a <_IO_stdin_used+0x14a>
    26c5:	ba 18 00 00 00       	mov    $0x18,%edx
    26ca:	be 01 00 00 00       	mov    $0x1,%esi
    26cf:	e8 9c ed ff ff       	callq  1470 <fwrite@plt>
    26d4:	31 ff                	xor    %edi,%edi
    26d6:	e8 85 ed ff ff       	callq  1460 <exit@plt>
    26db:	48 8b 0d 5e ea 00 00 	mov    0xea5e(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    26e2:	48 8d 3d 5f 7e 00 00 	lea    0x7e5f(%rip),%rdi        # a548 <_IO_stdin_used+0x548>
    26e9:	ba a1 00 00 00       	mov    $0xa1,%edx
    26ee:	be 01 00 00 00       	mov    $0x1,%esi
    26f3:	e8 78 ed ff ff       	callq  1470 <fwrite@plt>
    26f8:	31 ff                	xor    %edi,%edi
    26fa:	e8 61 ed ff ff       	callq  1460 <exit@plt>
    26ff:	83 bc 24 90 00 00 00 	cmpl   $0x8,0x90(%rsp)
    2706:	08 
    2707:	0f 8f 2d fe ff ff    	jg     253a <main+0x107a>
    270d:	c7 44 24 60 06 00 00 	movl   $0x6,0x60(%rsp)
    2714:	00 
    2715:	e9 b4 f0 ff ff       	jmpq   17ce <main+0x30e>
    271a:	48 89 bc 24 a8 00 00 	mov    %rdi,0xa8(%rsp)
    2721:	00 
    2722:	e9 55 f0 ff ff       	jmpq   177c <main+0x2bc>
    2727:	48 8b 0d 12 ea 00 00 	mov    0xea12(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    272e:	48 8d 3d 0b 7f 00 00 	lea    0x7f0b(%rip),%rdi        # a640 <_IO_stdin_used+0x640>
    2735:	ba 43 00 00 00       	mov    $0x43,%edx
    273a:	be 01 00 00 00       	mov    $0x1,%esi
    273f:	e8 2c ed ff ff       	callq  1470 <fwrite@plt>
    2744:	bf 01 00 00 00       	mov    $0x1,%edi
    2749:	e8 12 ed ff ff       	callq  1460 <exit@plt>
    274e:	e8 bd eb ff ff       	callq  1310 <__stack_chk_fail@plt>
    2753:	48 8d 0d 7f 7f 00 00 	lea    0x7f7f(%rip),%rcx        # a6d9 <__PRETTY_FUNCTION__.5741>
    275a:	ba 4d 01 00 00       	mov    $0x14d,%edx
    275f:	48 8d 35 c5 78 00 00 	lea    0x78c5(%rip),%rsi        # a02b <_IO_stdin_used+0x2b>
    2766:	48 8d 3d 83 7e 00 00 	lea    0x7e83(%rip),%rdi        # a5f0 <_IO_stdin_used+0x5f0>
    276d:	e8 ce eb ff ff       	callq  1340 <__assert_fail@plt>
    2772:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    2779:	00 00 00 
    277c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000002780 <_start>:
    2780:	f3 0f 1e fa          	endbr64 
    2784:	31 ed                	xor    %ebp,%ebp
    2786:	49 89 d1             	mov    %rdx,%r9
    2789:	5e                   	pop    %rsi
    278a:	48 89 e2             	mov    %rsp,%rdx
    278d:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    2791:	50                   	push   %rax
    2792:	54                   	push   %rsp
    2793:	4c 8d 05 a6 74 00 00 	lea    0x74a6(%rip),%r8        # 9c40 <__libc_csu_fini>
    279a:	48 8d 0d 2f 74 00 00 	lea    0x742f(%rip),%rcx        # 9bd0 <__libc_csu_init>
    27a1:	48 8d 3d 18 ed ff ff 	lea    -0x12e8(%rip),%rdi        # 14c0 <main>
    27a8:	ff 15 32 b8 00 00    	callq  *0xb832(%rip)        # dfe0 <__libc_start_main@GLIBC_2.2.5>
    27ae:	f4                   	hlt    
    27af:	90                   	nop

00000000000027b0 <deregister_tm_clones>:
    27b0:	48 8d 3d 71 e9 00 00 	lea    0xe971(%rip),%rdi        # 11128 <__TMC_END__>
    27b7:	48 8d 05 6a e9 00 00 	lea    0xe96a(%rip),%rax        # 11128 <__TMC_END__>
    27be:	48 39 f8             	cmp    %rdi,%rax
    27c1:	74 15                	je     27d8 <deregister_tm_clones+0x28>
    27c3:	48 8b 05 0e b8 00 00 	mov    0xb80e(%rip),%rax        # dfd8 <_ITM_deregisterTMCloneTable>
    27ca:	48 85 c0             	test   %rax,%rax
    27cd:	74 09                	je     27d8 <deregister_tm_clones+0x28>
    27cf:	ff e0                	jmpq   *%rax
    27d1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    27d8:	c3                   	retq   
    27d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000027e0 <register_tm_clones>:
    27e0:	48 8d 3d 41 e9 00 00 	lea    0xe941(%rip),%rdi        # 11128 <__TMC_END__>
    27e7:	48 8d 35 3a e9 00 00 	lea    0xe93a(%rip),%rsi        # 11128 <__TMC_END__>
    27ee:	48 29 fe             	sub    %rdi,%rsi
    27f1:	48 89 f0             	mov    %rsi,%rax
    27f4:	48 c1 ee 3f          	shr    $0x3f,%rsi
    27f8:	48 c1 f8 03          	sar    $0x3,%rax
    27fc:	48 01 c6             	add    %rax,%rsi
    27ff:	48 d1 fe             	sar    %rsi
    2802:	74 14                	je     2818 <register_tm_clones+0x38>
    2804:	48 8b 05 e5 b7 00 00 	mov    0xb7e5(%rip),%rax        # dff0 <_ITM_registerTMCloneTable>
    280b:	48 85 c0             	test   %rax,%rax
    280e:	74 08                	je     2818 <register_tm_clones+0x38>
    2810:	ff e0                	jmpq   *%rax
    2812:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2818:	c3                   	retq   
    2819:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002820 <__do_global_dtors_aux>:
    2820:	f3 0f 1e fa          	endbr64 
    2824:	80 3d 1d e9 00 00 00 	cmpb   $0x0,0xe91d(%rip)        # 11148 <completed.8060>
    282b:	75 2b                	jne    2858 <__do_global_dtors_aux+0x38>
    282d:	55                   	push   %rbp
    282e:	48 83 3d c2 b7 00 00 	cmpq   $0x0,0xb7c2(%rip)        # dff8 <__cxa_finalize@GLIBC_2.2.5>
    2835:	00 
    2836:	48 89 e5             	mov    %rsp,%rbp
    2839:	74 0c                	je     2847 <__do_global_dtors_aux+0x27>
    283b:	48 8b 3d c6 b7 00 00 	mov    0xb7c6(%rip),%rdi        # e008 <__dso_handle>
    2842:	e8 29 ea ff ff       	callq  1270 <__cxa_finalize@plt>
    2847:	e8 64 ff ff ff       	callq  27b0 <deregister_tm_clones>
    284c:	c6 05 f5 e8 00 00 01 	movb   $0x1,0xe8f5(%rip)        # 11148 <completed.8060>
    2853:	5d                   	pop    %rbp
    2854:	c3                   	retq   
    2855:	0f 1f 00             	nopl   (%rax)
    2858:	c3                   	retq   
    2859:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002860 <frame_dummy>:
    2860:	f3 0f 1e fa          	endbr64 
    2864:	e9 77 ff ff ff       	jmpq   27e0 <register_tm_clones>
    2869:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002870 <ctrl_bs_handler>:
    2870:	f3 0f 1e fa          	endbr64 
    2874:	48 83 ec 18          	sub    $0x18,%rsp
    2878:	31 ff                	xor    %edi,%edi
    287a:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2881:	00 00 
    2883:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    2888:	31 c0                	xor    %eax,%eax
    288a:	e8 41 eb ff ff       	callq  13d0 <time@plt>
    288f:	48 89 e7             	mov    %rsp,%rdi
    2892:	48 89 04 24          	mov    %rax,(%rsp)
    2896:	e8 65 ea ff ff       	callq  1300 <ctime@plt>
    289b:	48 8b 3d 9e e8 00 00 	mov    0xe89e(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    28a2:	48 89 c1             	mov    %rax,%rcx
    28a5:	48 8d 15 58 77 00 00 	lea    0x7758(%rip),%rdx        # a004 <_IO_stdin_used+0x4>
    28ac:	be 01 00 00 00       	mov    $0x1,%esi
    28b1:	31 c0                	xor    %eax,%eax
    28b3:	e8 c8 eb ff ff       	callq  1480 <__fprintf_chk@plt>
    28b8:	48 8b 0d 81 e8 00 00 	mov    0xe881(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    28bf:	ba 24 00 00 00       	mov    $0x24,%edx
    28c4:	be 01 00 00 00       	mov    $0x1,%esi
    28c9:	48 8d 3d 88 79 00 00 	lea    0x7988(%rip),%rdi        # a258 <_IO_stdin_used+0x258>
    28d0:	e8 9b eb ff ff       	callq  1470 <fwrite@plt>
    28d5:	8b 0d 35 00 01 00    	mov    0x10035(%rip),%ecx        # 12910 <readins>
    28db:	48 8b 3d 5e e8 00 00 	mov    0xe85e(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    28e2:	48 8d 15 97 79 00 00 	lea    0x7997(%rip),%rdx        # a280 <_IO_stdin_used+0x280>
    28e9:	be 01 00 00 00       	mov    $0x1,%esi
    28ee:	31 c0                	xor    %eax,%eax
    28f0:	e8 8b eb ff ff       	callq  1480 <__fprintf_chk@plt>
    28f5:	8b 0d 19 00 01 00    	mov    0x10019(%rip),%ecx        # 12914 <n>
    28fb:	48 8b 3d 3e e8 00 00 	mov    0xe83e(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    2902:	48 8d 15 00 77 00 00 	lea    0x7700(%rip),%rdx        # a009 <_IO_stdin_used+0x9>
    2909:	be 01 00 00 00       	mov    $0x1,%esi
    290e:	31 c0                	xor    %eax,%eax
    2910:	e8 6b eb ff ff       	callq  1480 <__fprintf_chk@plt>
    2915:	8b 15 fd ff 00 00    	mov    0xfffd(%rip),%edx        # 12918 <method>
    291b:	48 8d 05 fe b6 00 00 	lea    0xb6fe(%rip),%rax        # e020 <Methods>
    2922:	48 8b 0c d0          	mov    (%rax,%rdx,8),%rcx
    2926:	48 8b 3d 13 e8 00 00 	mov    0xe813(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    292d:	48 8d 15 ea 76 00 00 	lea    0x76ea(%rip),%rdx        # a01e <_IO_stdin_used+0x1e>
    2934:	be 01 00 00 00       	mov    $0x1,%esi
    2939:	31 c0                	xor    %eax,%eax
    293b:	e8 40 eb ff ff       	callq  1480 <__fprintf_chk@plt>
    2940:	48 8d 35 29 ff ff ff 	lea    -0xd7(%rip),%rsi        # 2870 <ctrl_bs_handler>
    2947:	bf 03 00 00 00       	mov    $0x3,%edi
    294c:	e8 4f ea ff ff       	callq  13a0 <signal@plt>
    2951:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    2956:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    295d:	00 00 
    295f:	75 05                	jne    2966 <ctrl_bs_handler+0xf6>
    2961:	48 83 c4 18          	add    $0x18,%rsp
    2965:	c3                   	retq   
    2966:	e8 a5 e9 ff ff       	callq  1310 <__stack_chk_fail@plt>
    296b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002970 <jfread>:
    2970:	f3 0f 1e fa          	endbr64 
    2974:	41 54                	push   %r12
    2976:	48 63 f6             	movslq %esi,%rsi
    2979:	55                   	push   %rbp
    297a:	53                   	push   %rbx
    297b:	48 85 c9             	test   %rcx,%rcx
    297e:	74 10                	je     2990 <jfread+0x20>
	return __fread_chk (__ptr, __bos0 (__ptr), __size, __n, __stream);

      if (__size * __n > __bos0 (__ptr))
	return __fread_chk_warn (__ptr, __bos0 (__ptr), __size, __n, __stream);
    }
  return __fread_alias (__ptr, __size, __n, __stream);
    2980:	e8 4b e9 ff ff       	callq  12d0 <fread@plt>
    2985:	5b                   	pop    %rbx
    2986:	5d                   	pop    %rbp
    2987:	41 5c                	pop    %r12
    2989:	c3                   	retq   
    298a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2990:	48 89 f5             	mov    %rsi,%rbp
    2993:	48 c1 ee 02          	shr    $0x2,%rsi
    2997:	85 f6                	test   %esi,%esi
    2999:	7e 25                	jle    29c0 <jfread+0x50>
    299b:	8d 46 ff             	lea    -0x1(%rsi),%eax
    299e:	48 89 fb             	mov    %rdi,%rbx
    29a1:	4c 8d 64 87 04       	lea    0x4(%rdi,%rax,4),%r12
    29a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    29ad:	00 00 00 
    29b0:	e8 bb e9 ff ff       	callq  1370 <mrand48@plt>
    29b5:	89 03                	mov    %eax,(%rbx)
    29b7:	48 83 c3 04          	add    $0x4,%rbx
    29bb:	49 39 dc             	cmp    %rbx,%r12
    29be:	75 f0                	jne    29b0 <jfread+0x40>
    29c0:	5b                   	pop    %rbx
    29c1:	89 e8                	mov    %ebp,%eax
    29c3:	5d                   	pop    %rbp
    29c4:	41 5c                	pop    %r12
    29c6:	c3                   	retq   
    29c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    29ce:	00 00 

00000000000029d0 <is_prime>:
    29d0:	f3 0f 1e fa          	endbr64 
    29d4:	48 81 ec f8 00 00 00 	sub    $0xf8,%rsp
    29db:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    29e2:	00 00 
    29e4:	48 89 84 24 e8 00 00 	mov    %rax,0xe8(%rsp)
    29eb:	00 
    29ec:	31 c0                	xor    %eax,%eax
    29ee:	48 b8 03 00 00 00 05 	movabs $0x500000003,%rax
    29f5:	00 00 00 
    29f8:	48 89 44 24 04       	mov    %rax,0x4(%rsp)
    29fd:	48 b8 07 00 00 00 0b 	movabs $0xb00000007,%rax
    2a04:	00 00 00 
    2a07:	48 89 44 24 0c       	mov    %rax,0xc(%rsp)
    2a0c:	48 b8 0d 00 00 00 11 	movabs $0x110000000d,%rax
    2a13:	00 00 00 
    2a16:	48 89 44 24 14       	mov    %rax,0x14(%rsp)
    2a1b:	48 b8 13 00 00 00 17 	movabs $0x1700000013,%rax
    2a22:	00 00 00 
    2a25:	48 89 44 24 1c       	mov    %rax,0x1c(%rsp)
    2a2a:	48 b8 1d 00 00 00 1f 	movabs $0x1f0000001d,%rax
    2a31:	00 00 00 
    2a34:	48 89 44 24 24       	mov    %rax,0x24(%rsp)
    2a39:	48 b8 25 00 00 00 29 	movabs $0x2900000025,%rax
    2a40:	00 00 00 
    2a43:	48 89 44 24 2c       	mov    %rax,0x2c(%rsp)
    2a48:	48 b8 2b 00 00 00 2f 	movabs $0x2f0000002b,%rax
    2a4f:	00 00 00 
    2a52:	48 89 44 24 34       	mov    %rax,0x34(%rsp)
    2a57:	48 b8 35 00 00 00 3b 	movabs $0x3b00000035,%rax
    2a5e:	00 00 00 
    2a61:	48 89 44 24 3c       	mov    %rax,0x3c(%rsp)
    2a66:	48 b8 3d 00 00 00 43 	movabs $0x430000003d,%rax
    2a6d:	00 00 00 
    2a70:	48 89 44 24 44       	mov    %rax,0x44(%rsp)
    2a75:	48 b8 47 00 00 00 49 	movabs $0x4900000047,%rax
    2a7c:	00 00 00 
    2a7f:	48 89 44 24 4c       	mov    %rax,0x4c(%rsp)
    2a84:	48 b8 4f 00 00 00 53 	movabs $0x530000004f,%rax
    2a8b:	00 00 00 
    2a8e:	48 89 44 24 54       	mov    %rax,0x54(%rsp)
    2a93:	48 b8 59 00 00 00 61 	movabs $0x6100000059,%rax
    2a9a:	00 00 00 
    2a9d:	48 89 44 24 5c       	mov    %rax,0x5c(%rsp)
    2aa2:	48 b8 65 00 00 00 67 	movabs $0x6700000065,%rax
    2aa9:	00 00 00 
    2aac:	48 89 44 24 64       	mov    %rax,0x64(%rsp)
    2ab1:	48 b8 6b 00 00 00 6d 	movabs $0x6d0000006b,%rax
    2ab8:	00 00 00 
    2abb:	48 89 44 24 6c       	mov    %rax,0x6c(%rsp)
    2ac0:	48 b8 71 00 00 00 7f 	movabs $0x7f00000071,%rax
    2ac7:	00 00 00 
    2aca:	48 89 44 24 74       	mov    %rax,0x74(%rsp)
    2acf:	48 b8 83 00 00 00 89 	movabs $0x8900000083,%rax
    2ad6:	00 00 00 
    2ad9:	48 89 44 24 7c       	mov    %rax,0x7c(%rsp)
    2ade:	48 b8 8b 00 00 00 95 	movabs $0x950000008b,%rax
    2ae5:	00 00 00 
    2ae8:	48 89 84 24 84 00 00 	mov    %rax,0x84(%rsp)
    2aef:	00 
    2af0:	48 b8 97 00 00 00 9d 	movabs $0x9d00000097,%rax
    2af7:	00 00 00 
    2afa:	48 89 84 24 8c 00 00 	mov    %rax,0x8c(%rsp)
    2b01:	00 
    2b02:	48 b8 a3 00 00 00 a7 	movabs $0xa7000000a3,%rax
    2b09:	00 00 00 
    2b0c:	48 89 84 24 94 00 00 	mov    %rax,0x94(%rsp)
    2b13:	00 
    2b14:	48 b8 ad 00 00 00 b3 	movabs $0xb3000000ad,%rax
    2b1b:	00 00 00 
    2b1e:	48 89 84 24 9c 00 00 	mov    %rax,0x9c(%rsp)
    2b25:	00 
    2b26:	48 b8 b5 00 00 00 bf 	movabs $0xbf000000b5,%rax
    2b2d:	00 00 00 
    2b30:	48 89 84 24 a4 00 00 	mov    %rax,0xa4(%rsp)
    2b37:	00 
    2b38:	48 b8 c1 00 00 00 c5 	movabs $0xc5000000c1,%rax
    2b3f:	00 00 00 
    2b42:	48 89 84 24 ac 00 00 	mov    %rax,0xac(%rsp)
    2b49:	00 
    2b4a:	48 b8 c7 00 00 00 d3 	movabs $0xd3000000c7,%rax
    2b51:	00 00 00 
    2b54:	48 89 84 24 b4 00 00 	mov    %rax,0xb4(%rsp)
    2b5b:	00 
    2b5c:	48 b8 df 00 00 00 e3 	movabs $0xe3000000df,%rax
    2b63:	00 00 00 
    2b66:	48 89 84 24 bc 00 00 	mov    %rax,0xbc(%rsp)
    2b6d:	00 
    2b6e:	48 b8 e5 00 00 00 e9 	movabs $0xe9000000e5,%rax
    2b75:	00 00 00 
    2b78:	48 89 84 24 c4 00 00 	mov    %rax,0xc4(%rsp)
    2b7f:	00 
    2b80:	48 b8 ef 00 00 00 f1 	movabs $0xf1000000ef,%rax
    2b87:	00 00 00 
    2b8a:	48 89 84 24 cc 00 00 	mov    %rax,0xcc(%rsp)
    2b91:	00 
    2b92:	48 b8 fb 00 00 00 01 	movabs $0x101000000fb,%rax
    2b99:	01 00 00 
    2b9c:	48 89 84 24 d4 00 00 	mov    %rax,0xd4(%rsp)
    2ba3:	00 
    2ba4:	48 8d 4c 24 04       	lea    0x4(%rsp),%rcx
    2ba9:	4c 8d 84 24 dc 00 00 	lea    0xdc(%rsp),%r8
    2bb0:	00 
    2bb1:	be 02 00 00 00       	mov    $0x2,%esi
    2bb6:	eb 13                	jmp    2bcb <is_prime+0x1fb>
    2bb8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2bbf:	00 
    2bc0:	4c 39 c1             	cmp    %r8,%rcx
    2bc3:	74 3b                	je     2c00 <is_prime+0x230>
    2bc5:	8b 31                	mov    (%rcx),%esi
    2bc7:	48 83 c1 04          	add    $0x4,%rcx
    2bcb:	89 f8                	mov    %edi,%eax
    2bcd:	99                   	cltd   
    2bce:	f7 fe                	idiv   %esi
    2bd0:	85 d2                	test   %edx,%edx
    2bd2:	75 ec                	jne    2bc0 <is_prime+0x1f0>
    2bd4:	39 f7                	cmp    %esi,%edi
    2bd6:	0f 94 c0             	sete   %al
    2bd9:	48 8b bc 24 e8 00 00 	mov    0xe8(%rsp),%rdi
    2be0:	00 
    2be1:	64 48 33 3c 25 28 00 	xor    %fs:0x28,%rdi
    2be8:	00 00 
    2bea:	75 33                	jne    2c1f <is_prime+0x24f>
    2bec:	0f b6 c0             	movzbl %al,%eax
    2bef:	48 81 c4 f8 00 00 00 	add    $0xf8,%rsp
    2bf6:	c3                   	retq   
    2bf7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    2bfe:	00 00 
    2c00:	48 8d 0d c9 7a 00 00 	lea    0x7ac9(%rip),%rcx        # a6d0 <__PRETTY_FUNCTION__.5802>
    2c07:	ba 6c 02 00 00       	mov    $0x26c,%edx
    2c0c:	48 8d 35 18 74 00 00 	lea    0x7418(%rip),%rsi        # a02b <_IO_stdin_used+0x2b>
    2c13:	48 8d 3d 1b 74 00 00 	lea    0x741b(%rip),%rdi        # a035 <_IO_stdin_used+0x35>
    2c1a:	e8 21 e7 ff ff       	callq  1340 <__assert_fail@plt>
    2c1f:	e8 ec e6 ff ff       	callq  1310 <__stack_chk_fail@plt>
    2c24:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    2c2b:	00 00 00 
    2c2e:	66 90                	xchg   %ax,%ax

0000000000002c30 <liberation_coding_bitmatrix>:
    2c30:	f3 0f 1e fa          	endbr64 
    2c34:	39 f7                	cmp    %esi,%edi
    2c36:	0f 8f 45 01 00 00    	jg     2d81 <liberation_coding_bitmatrix+0x151>
    2c3c:	41 56                	push   %r14
    2c3e:	41 55                	push   %r13
    2c40:	41 54                	push   %r12
    2c42:	41 89 fc             	mov    %edi,%r12d
    2c45:	44 0f af e6          	imul   %esi,%r12d
    2c49:	55                   	push   %rbp
    2c4a:	89 fd                	mov    %edi,%ebp
    2c4c:	45 89 e6             	mov    %r12d,%r14d
    2c4f:	44 0f af f6          	imul   %esi,%r14d
    2c53:	53                   	push   %rbx
    2c54:	89 f3                	mov    %esi,%ebx
    2c56:	43 8d 3c 36          	lea    (%r14,%r14,1),%edi
    2c5a:	48 63 ff             	movslq %edi,%rdi
    2c5d:	48 c1 e7 02          	shl    $0x2,%rdi
    2c61:	e8 9a e7 ff ff       	callq  1400 <malloc@plt>
    2c66:	49 89 c0             	mov    %rax,%r8
    2c69:	48 85 c0             	test   %rax,%rax
    2c6c:	0f 84 00 01 00 00    	je     2d72 <liberation_coding_bitmatrix+0x142>
    2c72:	4c 63 eb             	movslq %ebx,%r13
    2c75:	4c 89 ea             	mov    %r13,%rdx
    2c78:	49 0f af d5          	imul   %r13,%rdx
    2c7c:	48 63 c5             	movslq %ebp,%rax
  (void) __builtin___memset_chk (__dest, '\0', __len, __bos0 (__dest));
    2c7f:	4c 89 c7             	mov    %r8,%rdi
    2c82:	48 0f af d0          	imul   %rax,%rdx
    2c86:	31 f6                	xor    %esi,%esi
    2c88:	48 c1 e2 03          	shl    $0x3,%rdx
    2c8c:	e8 bf e6 ff ff       	callq  1350 <memset@plt>
    2c91:	49 89 c0             	mov    %rax,%r8
    2c94:	85 db                	test   %ebx,%ebx
    2c96:	7e 40                	jle    2cd8 <liberation_coding_bitmatrix+0xa8>
    2c98:	49 63 c4             	movslq %r12d,%rax
    2c9b:	4c 8d 0c 85 04 00 00 	lea    0x4(,%rax,4),%r9
    2ca2:	00 
    2ca3:	4c 89 c7             	mov    %r8,%rdi
    2ca6:	4a 8d 0c ad 00 00 00 	lea    0x0(,%r13,4),%rcx
    2cad:	00 
    2cae:	31 f6                	xor    %esi,%esi
    2cb0:	85 ed                	test   %ebp,%ebp
    2cb2:	7e 1b                	jle    2ccf <liberation_coding_bitmatrix+0x9f>
    2cb4:	48 89 fa             	mov    %rdi,%rdx
    2cb7:	31 c0                	xor    %eax,%eax
    2cb9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2cc0:	ff c0                	inc    %eax
    2cc2:	c7 02 01 00 00 00    	movl   $0x1,(%rdx)
    2cc8:	48 01 ca             	add    %rcx,%rdx
    2ccb:	39 c5                	cmp    %eax,%ebp
    2ccd:	75 f1                	jne    2cc0 <liberation_coding_bitmatrix+0x90>
    2ccf:	ff c6                	inc    %esi
    2cd1:	4c 01 cf             	add    %r9,%rdi
    2cd4:	39 f3                	cmp    %esi,%ebx
    2cd6:	75 d8                	jne    2cb0 <liberation_coding_bitmatrix+0x80>
    2cd8:	85 ed                	test   %ebp,%ebp
    2cda:	0f 8e 86 00 00 00    	jle    2d66 <liberation_coding_bitmatrix+0x136>
    2ce0:	8d 43 ff             	lea    -0x1(%rbx),%eax
    2ce3:	41 89 c5             	mov    %eax,%r13d
    2ce6:	41 c1 ed 1f          	shr    $0x1f,%r13d
    2cea:	41 01 c5             	add    %eax,%r13d
    2ced:	41 d1 fd             	sar    %r13d
    2cf0:	45 89 f2             	mov    %r14d,%r10d
    2cf3:	89 df                	mov    %ebx,%edi
    2cf5:	45 31 db             	xor    %r11d,%r11d
    2cf8:	45 31 c9             	xor    %r9d,%r9d
    2cfb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    2d00:	85 db                	test   %ebx,%ebx
    2d02:	7e 27                	jle    2d2b <liberation_coding_bitmatrix+0xfb>
    2d04:	44 89 c9             	mov    %r9d,%ecx
    2d07:	44 89 d6             	mov    %r10d,%esi
    2d0a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2d10:	89 c8                	mov    %ecx,%eax
    2d12:	99                   	cltd   
    2d13:	f7 fb                	idiv   %ebx
    2d15:	ff c1                	inc    %ecx
    2d17:	01 f2                	add    %esi,%edx
    2d19:	48 63 d2             	movslq %edx,%rdx
    2d1c:	41 c7 04 90 01 00 00 	movl   $0x1,(%r8,%rdx,4)
    2d23:	00 
    2d24:	44 01 e6             	add    %r12d,%esi
    2d27:	39 cf                	cmp    %ecx,%edi
    2d29:	75 e5                	jne    2d10 <liberation_coding_bitmatrix+0xe0>
    2d2b:	45 85 c9             	test   %r9d,%r9d
    2d2e:	74 26                	je     2d56 <liberation_coding_bitmatrix+0x126>
    2d30:	44 89 d8             	mov    %r11d,%eax
    2d33:	99                   	cltd   
    2d34:	f7 fb                	idiv   %ebx
    2d36:	89 e9                	mov    %ebp,%ecx
    2d38:	41 8d 44 11 ff       	lea    -0x1(%r9,%rdx,1),%eax
    2d3d:	0f af ca             	imul   %edx,%ecx
    2d40:	99                   	cltd   
    2d41:	f7 fb                	idiv   %ebx
    2d43:	0f af cb             	imul   %ebx,%ecx
    2d46:	44 01 d1             	add    %r10d,%ecx
    2d49:	01 d1                	add    %edx,%ecx
    2d4b:	48 63 c9             	movslq %ecx,%rcx
    2d4e:	41 c7 04 88 01 00 00 	movl   $0x1,(%r8,%rcx,4)
    2d55:	00 
    2d56:	41 ff c1             	inc    %r9d
    2d59:	45 01 eb             	add    %r13d,%r11d
    2d5c:	41 01 da             	add    %ebx,%r10d
    2d5f:	ff c7                	inc    %edi
    2d61:	44 39 cd             	cmp    %r9d,%ebp
    2d64:	75 9a                	jne    2d00 <liberation_coding_bitmatrix+0xd0>
    2d66:	5b                   	pop    %rbx
    2d67:	5d                   	pop    %rbp
    2d68:	41 5c                	pop    %r12
    2d6a:	41 5d                	pop    %r13
    2d6c:	4c 89 c0             	mov    %r8,%rax
    2d6f:	41 5e                	pop    %r14
    2d71:	c3                   	retq   
    2d72:	5b                   	pop    %rbx
    2d73:	5d                   	pop    %rbp
    2d74:	41 5c                	pop    %r12
    2d76:	45 31 c0             	xor    %r8d,%r8d
    2d79:	41 5d                	pop    %r13
    2d7b:	4c 89 c0             	mov    %r8,%rax
    2d7e:	41 5e                	pop    %r14
    2d80:	c3                   	retq   
    2d81:	45 31 c0             	xor    %r8d,%r8d
    2d84:	4c 89 c0             	mov    %r8,%rax
    2d87:	c3                   	retq   
    2d88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2d8f:	00 

0000000000002d90 <liber8tion_coding_bitmatrix>:
    2d90:	f3 0f 1e fa          	endbr64 
    2d94:	83 ff 08             	cmp    $0x8,%edi
    2d97:	0f 8f 22 04 00 00    	jg     31bf <liber8tion_coding_bitmatrix+0x42f>
    2d9d:	41 54                	push   %r12
    2d9f:	be 01 00 00 00       	mov    $0x1,%esi
    2da4:	55                   	push   %rbp
    2da5:	53                   	push   %rbx
    2da6:	89 fb                	mov    %edi,%ebx
    2da8:	c1 e7 07             	shl    $0x7,%edi
    2dab:	48 63 ff             	movslq %edi,%rdi
    2dae:	48 c1 e7 02          	shl    $0x2,%rdi
    2db2:	e8 c9 e5 ff ff       	callq  1380 <calloc@plt>
    2db7:	48 85 c0             	test   %rax,%rax
    2dba:	0f 84 f8 03 00 00    	je     31b8 <liber8tion_coding_bitmatrix+0x428>
    2dc0:	8d 7b ff             	lea    -0x1(%rbx),%edi
    2dc3:	48 ff c7             	inc    %rdi
    2dc6:	48 89 f9             	mov    %rdi,%rcx
    2dc9:	44 8d 0c dd 00 00 00 	lea    0x0(,%rbx,8),%r9d
    2dd0:	00 
    2dd1:	49 63 d1             	movslq %r9d,%rdx
    2dd4:	48 c1 e1 05          	shl    $0x5,%rcx
    2dd8:	48 f7 df             	neg    %rdi
    2ddb:	4c 8d 04 95 04 00 00 	lea    0x4(,%rdx,4),%r8
    2de2:	00 
    2de3:	48 01 c1             	add    %rax,%rcx
    2de6:	be 08 00 00 00       	mov    $0x8,%esi
    2deb:	48 c1 e7 05          	shl    $0x5,%rdi
    2def:	90                   	nop
    2df0:	85 db                	test   %ebx,%ebx
    2df2:	7e 1b                	jle    2e0f <liber8tion_coding_bitmatrix+0x7f>
    2df4:	48 8d 14 0f          	lea    (%rdi,%rcx,1),%rdx
    2df8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2dff:	00 
    2e00:	c7 02 01 00 00 00    	movl   $0x1,(%rdx)
    2e06:	48 83 c2 20          	add    $0x20,%rdx
    2e0a:	48 39 ca             	cmp    %rcx,%rdx
    2e0d:	75 f1                	jne    2e00 <liber8tion_coding_bitmatrix+0x70>
    2e0f:	4c 01 c1             	add    %r8,%rcx
    2e12:	ff ce                	dec    %esi
    2e14:	75 da                	jne    2df0 <liber8tion_coding_bitmatrix+0x60>
    2e16:	85 db                	test   %ebx,%ebx
    2e18:	0f 84 95 03 00 00    	je     31b3 <liber8tion_coding_bitmatrix+0x423>
    2e1e:	41 89 d8             	mov    %ebx,%r8d
    2e21:	41 c1 e0 06          	shl    $0x6,%r8d
    2e25:	43 8d 3c 08          	lea    (%r8,%r9,1),%edi
    2e29:	49 63 d0             	movslq %r8d,%rdx
    2e2c:	c7 04 90 01 00 00 00 	movl   $0x1,(%rax,%rdx,4)
    2e33:	42 8d 2c 0f          	lea    (%rdi,%r9,1),%ebp
    2e37:	48 63 d7             	movslq %edi,%rdx
    2e3a:	c7 44 90 04 01 00 00 	movl   $0x1,0x4(%rax,%rdx,4)
    2e41:	00 
    2e42:	46 8d 54 0d 00       	lea    0x0(%rbp,%r9,1),%r10d
    2e47:	48 63 d5             	movslq %ebp,%rdx
    2e4a:	c7 44 90 08 01 00 00 	movl   $0x1,0x8(%rax,%rdx,4)
    2e51:	00 
    2e52:	43 8d 34 0a          	lea    (%r10,%r9,1),%esi
    2e56:	49 63 d2             	movslq %r10d,%rdx
    2e59:	c7 44 90 0c 01 00 00 	movl   $0x1,0xc(%rax,%rdx,4)
    2e60:	00 
    2e61:	42 8d 0c 0e          	lea    (%rsi,%r9,1),%ecx
    2e65:	48 63 d6             	movslq %esi,%rdx
    2e68:	c7 44 90 10 01 00 00 	movl   $0x1,0x10(%rax,%rdx,4)
    2e6f:	00 
    2e70:	48 63 d1             	movslq %ecx,%rdx
    2e73:	c7 44 90 14 01 00 00 	movl   $0x1,0x14(%rax,%rdx,4)
    2e7a:	00 
    2e7b:	42 8d 14 09          	lea    (%rcx,%r9,1),%edx
    2e7f:	4c 63 da             	movslq %edx,%r11
    2e82:	41 01 d1             	add    %edx,%r9d
    2e85:	42 c7 44 98 18 01 00 	movl   $0x1,0x18(%rax,%r11,4)
    2e8c:	00 00 
    2e8e:	4d 63 d9             	movslq %r9d,%r11
    2e91:	42 c7 44 98 1c 01 00 	movl   $0x1,0x1c(%rax,%r11,4)
    2e98:	00 00 
    2e9a:	83 fb 01             	cmp    $0x1,%ebx
    2e9d:	0f 84 10 03 00 00    	je     31b3 <liber8tion_coding_bitmatrix+0x423>
    2ea3:	41 83 c0 08          	add    $0x8,%r8d
    2ea7:	4d 63 c0             	movslq %r8d,%r8
    2eaa:	49 83 c0 07          	add    $0x7,%r8
    2eae:	42 c7 04 80 01 00 00 	movl   $0x1,(%rax,%r8,4)
    2eb5:	00 
    2eb6:	4e 8d 1c 85 00 00 00 	lea    0x0(,%r8,4),%r11
    2ebd:	00 
    2ebe:	44 8d 47 08          	lea    0x8(%rdi),%r8d
    2ec2:	4d 63 c0             	movslq %r8d,%r8
    2ec5:	83 c5 08             	add    $0x8,%ebp
    2ec8:	49 83 c0 03          	add    $0x3,%r8
    2ecc:	48 63 ed             	movslq %ebp,%rbp
    2ecf:	42 c7 04 80 01 00 00 	movl   $0x1,(%rax,%r8,4)
    2ed6:	00 
    2ed7:	4a 8d 3c 85 00 00 00 	lea    0x0(,%r8,4),%rdi
    2ede:	00 
    2edf:	c7 04 a8 01 00 00 00 	movl   $0x1,(%rax,%rbp,4)
    2ee6:	4c 8d 04 ad 00 00 00 	lea    0x0(,%rbp,4),%r8
    2eed:	00 
    2eee:	41 8d 6a 08          	lea    0x8(%r10),%ebp
    2ef2:	48 63 ed             	movslq %ebp,%rbp
    2ef5:	48 83 c5 02          	add    $0x2,%rbp
    2ef9:	c7 04 a8 01 00 00 00 	movl   $0x1,(%rax,%rbp,4)
    2f00:	4c 8d 14 ad 00 00 00 	lea    0x0(,%rbp,4),%r10
    2f07:	00 
    2f08:	83 c1 08             	add    $0x8,%ecx
    2f0b:	8d 6e 08             	lea    0x8(%rsi),%ebp
    2f0e:	83 c2 08             	add    $0x8,%edx
    2f11:	48 63 ed             	movslq %ebp,%rbp
    2f14:	48 63 c9             	movslq %ecx,%rcx
    2f17:	48 63 d2             	movslq %edx,%rdx
    2f1a:	48 83 c5 06          	add    $0x6,%rbp
    2f1e:	48 ff c1             	inc    %rcx
    2f21:	48 83 c2 05          	add    $0x5,%rdx
    2f25:	c7 04 a8 01 00 00 00 	movl   $0x1,(%rax,%rbp,4)
    2f2c:	4c 8d 24 95 00 00 00 	lea    0x0(,%rdx,4),%r12
    2f33:	00 
    2f34:	c7 04 88 01 00 00 00 	movl   $0x1,(%rax,%rcx,4)
    2f3b:	c7 04 90 01 00 00 00 	movl   $0x1,(%rax,%rdx,4)
    2f42:	41 8d 51 08          	lea    0x8(%r9),%edx
    2f46:	48 63 d2             	movslq %edx,%rdx
    2f49:	48 8d 34 ad 00 00 00 	lea    0x0(,%rbp,4),%rsi
    2f50:	00 
    2f51:	48 83 c2 04          	add    $0x4,%rdx
    2f55:	c7 04 90 01 00 00 00 	movl   $0x1,(%rax,%rdx,4)
    2f5c:	48 8d 2c 8d 00 00 00 	lea    0x0(,%rcx,4),%rbp
    2f63:	00 
    2f64:	c7 44 30 04 01 00 00 	movl   $0x1,0x4(%rax,%rsi,1)
    2f6b:	00 
    2f6c:	48 8d 0c 95 00 00 00 	lea    0x0(,%rdx,4),%rcx
    2f73:	00 
    2f74:	83 fb 02             	cmp    $0x2,%ebx
    2f77:	0f 84 36 02 00 00    	je     31b3 <liber8tion_coding_bitmatrix+0x423>
    2f7d:	42 c7 44 18 1c 01 00 	movl   $0x1,0x1c(%rax,%r11,1)
    2f84:	00 00 
    2f86:	c7 44 38 1c 01 00 00 	movl   $0x1,0x1c(%rax,%rdi,1)
    2f8d:	00 
    2f8e:	42 c7 44 00 30 01 00 	movl   $0x1,0x30(%rax,%r8,1)
    2f95:	00 00 
    2f97:	42 c7 44 10 18 01 00 	movl   $0x1,0x18(%rax,%r10,1)
    2f9e:	00 00 
    2fa0:	c7 44 30 24 01 00 00 	movl   $0x1,0x24(%rax,%rsi,1)
    2fa7:	00 
    2fa8:	c7 44 28 28 01 00 00 	movl   $0x1,0x28(%rax,%rbp,1)
    2faf:	00 
    2fb0:	42 c7 44 20 10 01 00 	movl   $0x1,0x10(%rax,%r12,1)
    2fb7:	00 00 
    2fb9:	c7 44 08 24 01 00 00 	movl   $0x1,0x24(%rax,%rcx,1)
    2fc0:	00 
    2fc1:	c7 44 38 20 01 00 00 	movl   $0x1,0x20(%rax,%rdi,1)
    2fc8:	00 
    2fc9:	83 fb 03             	cmp    $0x3,%ebx
    2fcc:	0f 84 e1 01 00 00    	je     31b3 <liber8tion_coding_bitmatrix+0x423>
    2fd2:	42 c7 44 18 2c 01 00 	movl   $0x1,0x2c(%rax,%r11,1)
    2fd9:	00 00 
    2fdb:	c7 44 38 48 01 00 00 	movl   $0x1,0x48(%rax,%rdi,1)
    2fe2:	00 
    2fe3:	42 c7 44 00 5c 01 00 	movl   $0x1,0x5c(%rax,%r8,1)
    2fea:	00 00 
    2fec:	42 c7 44 10 50 01 00 	movl   $0x1,0x50(%rax,%r10,1)
    2ff3:	00 00 
    2ff5:	c7 44 30 28 01 00 00 	movl   $0x1,0x28(%rax,%rsi,1)
    2ffc:	00 
    2ffd:	c7 44 28 48 01 00 00 	movl   $0x1,0x48(%rax,%rbp,1)
    3004:	00 
    3005:	42 c7 44 20 3c 01 00 	movl   $0x1,0x3c(%rax,%r12,1)
    300c:	00 00 
    300e:	c7 44 08 34 01 00 00 	movl   $0x1,0x34(%rax,%rcx,1)
    3015:	00 
    3016:	c7 44 28 4c 01 00 00 	movl   $0x1,0x4c(%rax,%rbp,1)
    301d:	00 
    301e:	83 fb 04             	cmp    $0x4,%ebx
    3021:	0f 84 8c 01 00 00    	je     31b3 <liber8tion_coding_bitmatrix+0x423>
    3027:	42 c7 44 18 58 01 00 	movl   $0x1,0x58(%rax,%r11,1)
    302e:	00 00 
    3030:	c7 44 38 6c 01 00 00 	movl   $0x1,0x6c(%rax,%rdi,1)
    3037:	00 
    3038:	42 c7 44 00 64 01 00 	movl   $0x1,0x64(%rax,%r8,1)
    303f:	00 00 
    3041:	42 c7 44 10 74 01 00 	movl   $0x1,0x74(%rax,%r10,1)
    3048:	00 00 
    304a:	c7 44 30 50 01 00 00 	movl   $0x1,0x50(%rax,%rsi,1)
    3051:	00 
    3052:	c7 44 28 6c 01 00 00 	movl   $0x1,0x6c(%rax,%rbp,1)
    3059:	00 
    305a:	42 c7 44 20 58 01 00 	movl   $0x1,0x58(%rax,%r12,1)
    3061:	00 00 
    3063:	c7 44 08 50 01 00 00 	movl   $0x1,0x50(%rax,%rcx,1)
    306a:	00 
    306b:	42 c7 44 00 60 01 00 	movl   $0x1,0x60(%rax,%r8,1)
    3072:	00 00 
    3074:	83 fb 05             	cmp    $0x5,%ebx
    3077:	0f 84 36 01 00 00    	je     31b3 <liber8tion_coding_bitmatrix+0x423>
    307d:	42 c7 44 18 68 01 00 	movl   $0x1,0x68(%rax,%r11,1)
    3084:	00 00 
    3086:	c7 44 38 7c 01 00 00 	movl   $0x1,0x7c(%rax,%rdi,1)
    308d:	00 
    308e:	42 c7 84 00 8c 00 00 	movl   $0x1,0x8c(%rax,%r8,1)
    3095:	00 01 00 00 00 
    309a:	42 c7 84 10 88 00 00 	movl   $0x1,0x88(%rax,%r10,1)
    30a1:	00 01 00 00 00 
    30a6:	c7 44 30 7c 01 00 00 	movl   $0x1,0x7c(%rax,%rsi,1)
    30ad:	00 
    30ae:	c7 84 28 94 00 00 00 	movl   $0x1,0x94(%rax,%rbp,1)
    30b5:	01 00 00 00 
    30b9:	42 c7 84 20 88 00 00 	movl   $0x1,0x88(%rax,%r12,1)
    30c0:	00 01 00 00 00 
    30c5:	c7 44 08 70 01 00 00 	movl   $0x1,0x70(%rax,%rcx,1)
    30cc:	00 
    30cd:	c7 44 08 78 01 00 00 	movl   $0x1,0x78(%rax,%rcx,1)
    30d4:	00 
    30d5:	83 fb 06             	cmp    $0x6,%ebx
    30d8:	0f 84 d5 00 00 00    	je     31b3 <liber8tion_coding_bitmatrix+0x423>
    30de:	42 c7 84 18 90 00 00 	movl   $0x1,0x90(%rax,%r11,1)
    30e5:	00 01 00 00 00 
    30ea:	c7 84 38 94 00 00 00 	movl   $0x1,0x94(%rax,%rdi,1)
    30f1:	01 00 00 00 
    30f5:	42 c7 84 00 b8 00 00 	movl   $0x1,0xb8(%rax,%r8,1)
    30fc:	00 01 00 00 00 
    3101:	42 c7 84 10 ac 00 00 	movl   $0x1,0xac(%rax,%r10,1)
    3108:	00 01 00 00 00 
    310d:	c7 84 30 8c 00 00 00 	movl   $0x1,0x8c(%rax,%rsi,1)
    3114:	01 00 00 00 
    3118:	c7 84 28 b8 00 00 00 	movl   $0x1,0xb8(%rax,%rbp,1)
    311f:	01 00 00 00 
    3123:	42 c7 84 20 9c 00 00 	movl   $0x1,0x9c(%rax,%r12,1)
    312a:	00 01 00 00 00 
    312f:	c7 84 08 98 00 00 00 	movl   $0x1,0x98(%rax,%rcx,1)
    3136:	01 00 00 00 
    313a:	42 c7 84 20 a0 00 00 	movl   $0x1,0xa0(%rax,%r12,1)
    3141:	00 01 00 00 00 
    3146:	83 fb 07             	cmp    $0x7,%ebx
    3149:	74 68                	je     31b3 <liber8tion_coding_bitmatrix+0x423>
    314b:	42 c7 84 18 b4 00 00 	movl   $0x1,0xb4(%rax,%r11,1)
    3152:	00 01 00 00 00 
    3157:	c7 84 38 d0 00 00 00 	movl   $0x1,0xd0(%rax,%rdi,1)
    315e:	01 00 00 00 
    3162:	42 c7 84 00 c4 00 00 	movl   $0x1,0xc4(%rax,%r8,1)
    3169:	00 01 00 00 00 
    316e:	42 c7 84 10 cc 00 00 	movl   $0x1,0xcc(%rax,%r10,1)
    3175:	00 01 00 00 00 
    317a:	c7 84 30 b4 00 00 00 	movl   $0x1,0xb4(%rax,%rsi,1)
    3181:	01 00 00 00 
    3185:	c7 84 28 c4 00 00 00 	movl   $0x1,0xc4(%rax,%rbp,1)
    318c:	01 00 00 00 
    3190:	42 c7 84 20 ac 00 00 	movl   $0x1,0xac(%rax,%r12,1)
    3197:	00 01 00 00 00 
    319c:	c7 84 08 c8 00 00 00 	movl   $0x1,0xc8(%rax,%rcx,1)
    31a3:	01 00 00 00 
    31a7:	42 c7 84 10 bc 00 00 	movl   $0x1,0xbc(%rax,%r10,1)
    31ae:	00 01 00 00 00 
    31b3:	5b                   	pop    %rbx
    31b4:	5d                   	pop    %rbp
    31b5:	41 5c                	pop    %r12
    31b7:	c3                   	retq   
    31b8:	5b                   	pop    %rbx
    31b9:	5d                   	pop    %rbp
    31ba:	31 c0                	xor    %eax,%eax
    31bc:	41 5c                	pop    %r12
    31be:	c3                   	retq   
    31bf:	31 c0                	xor    %eax,%eax
    31c1:	c3                   	retq   
    31c2:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    31c9:	00 00 00 00 
    31cd:	0f 1f 00             	nopl   (%rax)

00000000000031d0 <blaum_roth_coding_bitmatrix>:
    31d0:	f3 0f 1e fa          	endbr64 
    31d4:	39 f7                	cmp    %esi,%edi
    31d6:	0f 8f dd 01 00 00    	jg     33b9 <blaum_roth_coding_bitmatrix+0x1e9>
    31dc:	41 57                	push   %r15
    31de:	41 56                	push   %r14
    31e0:	41 55                	push   %r13
    31e2:	41 54                	push   %r12
    31e4:	41 89 fc             	mov    %edi,%r12d
    31e7:	44 0f af e6          	imul   %esi,%r12d
    31eb:	55                   	push   %rbp
    31ec:	89 f5                	mov    %esi,%ebp
    31ee:	45 89 e5             	mov    %r12d,%r13d
    31f1:	44 0f af ee          	imul   %esi,%r13d
    31f5:	53                   	push   %rbx
    31f6:	89 fb                	mov    %edi,%ebx
    31f8:	43 8d 7c 2d 00       	lea    0x0(%r13,%r13,1),%edi
    31fd:	48 63 ff             	movslq %edi,%rdi
    3200:	48 83 ec 18          	sub    $0x18,%rsp
    3204:	48 c1 e7 02          	shl    $0x2,%rdi
    3208:	e8 f3 e1 ff ff       	callq  1400 <malloc@plt>
    320d:	49 89 c1             	mov    %rax,%r9
    3210:	48 85 c0             	test   %rax,%rax
    3213:	0f 84 9b 01 00 00    	je     33b4 <blaum_roth_coding_bitmatrix+0x1e4>
    3219:	4c 63 f5             	movslq %ebp,%r14
    321c:	4c 89 f2             	mov    %r14,%rdx
    321f:	49 0f af d6          	imul   %r14,%rdx
    3223:	48 63 c3             	movslq %ebx,%rax
    3226:	4c 89 cf             	mov    %r9,%rdi
    3229:	48 0f af d0          	imul   %rax,%rdx
    322d:	31 f6                	xor    %esi,%esi
    322f:	48 c1 e2 03          	shl    $0x3,%rdx
    3233:	e8 18 e1 ff ff       	callq  1350 <memset@plt>
    3238:	49 89 c1             	mov    %rax,%r9
    323b:	85 ed                	test   %ebp,%ebp
    323d:	7e 49                	jle    3288 <blaum_roth_coding_bitmatrix+0xb8>
    323f:	49 63 c4             	movslq %r12d,%rax
    3242:	4c 8d 04 85 04 00 00 	lea    0x4(,%rax,4),%r8
    3249:	00 
    324a:	4c 89 cf             	mov    %r9,%rdi
    324d:	4a 8d 0c b5 00 00 00 	lea    0x0(,%r14,4),%rcx
    3254:	00 
    3255:	31 f6                	xor    %esi,%esi
    3257:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    325e:	00 00 
    3260:	48 89 fa             	mov    %rdi,%rdx
    3263:	31 c0                	xor    %eax,%eax
    3265:	85 db                	test   %ebx,%ebx
    3267:	7e 16                	jle    327f <blaum_roth_coding_bitmatrix+0xaf>
    3269:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3270:	ff c0                	inc    %eax
    3272:	c7 02 01 00 00 00    	movl   $0x1,(%rdx)
    3278:	48 01 ca             	add    %rcx,%rdx
    327b:	39 c3                	cmp    %eax,%ebx
    327d:	75 f1                	jne    3270 <blaum_roth_coding_bitmatrix+0xa0>
    327f:	ff c6                	inc    %esi
    3281:	4c 01 c7             	add    %r8,%rdi
    3284:	39 f5                	cmp    %esi,%ebp
    3286:	75 d8                	jne    3260 <blaum_roth_coding_bitmatrix+0x90>
    3288:	44 8d 45 01          	lea    0x1(%rbp),%r8d
    328c:	85 db                	test   %ebx,%ebx
    328e:	0f 8e d4 00 00 00    	jle    3368 <blaum_roth_coding_bitmatrix+0x198>
    3294:	44 89 c0             	mov    %r8d,%eax
    3297:	c1 e8 1f             	shr    $0x1f,%eax
    329a:	44 01 c0             	add    %r8d,%eax
    329d:	d1 f8                	sar    %eax
    329f:	ff c0                	inc    %eax
    32a1:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    32a5:	4a 8d 04 b5 00 00 00 	lea    0x0(,%r14,4),%rax
    32ac:	00 
    32ad:	48 89 04 24          	mov    %rax,(%rsp)
    32b1:	49 63 c5             	movslq %r13d,%rax
    32b4:	4d 8d 3c 81          	lea    (%r9,%rax,4),%r15
    32b8:	49 63 c4             	movslq %r12d,%rax
    32bb:	44 89 c6             	mov    %r8d,%esi
    32be:	4c 8d 34 85 04 00 00 	lea    0x4(,%rax,4),%r14
    32c5:	00 
    32c6:	31 ff                	xor    %edi,%edi
    32c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    32cf:	00 
    32d0:	44 89 ea             	mov    %r13d,%edx
    32d3:	85 ff                	test   %edi,%edi
    32d5:	0f 84 a5 00 00 00    	je     3380 <blaum_roth_coding_bitmatrix+0x1b0>
    32db:	85 ed                	test   %ebp,%ebp
    32dd:	7e 76                	jle    3355 <blaum_roth_coding_bitmatrix+0x185>
    32df:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    32e3:	41 89 fb             	mov    %edi,%r11d
    32e6:	41 d1 fb             	sar    %r11d
    32e9:	44 01 d8             	add    %r11d,%eax
    32ec:	40 f6 c7 01          	test   $0x1,%dil
    32f0:	44 0f 45 d8          	cmovne %eax,%r11d
    32f4:	b8 01 00 00 00       	mov    $0x1,%eax
    32f9:	41 ff cb             	dec    %r11d
    32fc:	eb 2b                	jmp    3329 <blaum_roth_coding_bitmatrix+0x159>
    32fe:	66 90                	xchg   %ax,%ax
    3300:	8d 0c 38             	lea    (%rax,%rdi,1),%ecx
    3303:	41 89 c2             	mov    %eax,%r10d
    3306:	41 29 f2             	sub    %esi,%r10d
    3309:	44 39 c1             	cmp    %r8d,%ecx
    330c:	41 0f 4d ca          	cmovge %r10d,%ecx
    3310:	ff c0                	inc    %eax
    3312:	8d 4c 11 ff          	lea    -0x1(%rcx,%rdx,1),%ecx
    3316:	48 63 c9             	movslq %ecx,%rcx
    3319:	41 c7 04 89 01 00 00 	movl   $0x1,(%r9,%rcx,4)
    3320:	00 
    3321:	44 01 e2             	add    %r12d,%edx
    3324:	44 39 c0             	cmp    %r8d,%eax
    3327:	74 2c                	je     3355 <blaum_roth_coding_bitmatrix+0x185>
    3329:	39 f0                	cmp    %esi,%eax
    332b:	75 d3                	jne    3300 <blaum_roth_coding_bitmatrix+0x130>
    332d:	8d 0c 3a             	lea    (%rdx,%rdi,1),%ecx
    3330:	48 63 c9             	movslq %ecx,%rcx
    3333:	41 c7 44 89 fc 01 00 	movl   $0x1,-0x4(%r9,%rcx,4)
    333a:	00 00 
    333c:	41 8d 0c 13          	lea    (%r11,%rdx,1),%ecx
    3340:	48 63 c9             	movslq %ecx,%rcx
    3343:	ff c0                	inc    %eax
    3345:	41 c7 04 89 01 00 00 	movl   $0x1,(%r9,%rcx,4)
    334c:	00 
    334d:	44 01 e2             	add    %r12d,%edx
    3350:	44 39 c0             	cmp    %r8d,%eax
    3353:	75 d4                	jne    3329 <blaum_roth_coding_bitmatrix+0x159>
    3355:	ff c7                	inc    %edi
    3357:	41 01 ed             	add    %ebp,%r13d
    335a:	ff ce                	dec    %esi
    335c:	4c 03 3c 24          	add    (%rsp),%r15
    3360:	39 fb                	cmp    %edi,%ebx
    3362:	0f 85 68 ff ff ff    	jne    32d0 <blaum_roth_coding_bitmatrix+0x100>
    3368:	48 83 c4 18          	add    $0x18,%rsp
    336c:	5b                   	pop    %rbx
    336d:	5d                   	pop    %rbp
    336e:	41 5c                	pop    %r12
    3370:	41 5d                	pop    %r13
    3372:	41 5e                	pop    %r14
    3374:	4c 89 c8             	mov    %r9,%rax
    3377:	41 5f                	pop    %r15
    3379:	c3                   	retq   
    337a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    3380:	85 ed                	test   %ebp,%ebp
    3382:	7e d1                	jle    3355 <blaum_roth_coding_bitmatrix+0x185>
    3384:	4c 89 fa             	mov    %r15,%rdx
    3387:	31 c0                	xor    %eax,%eax
    3389:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3390:	ff c0                	inc    %eax
    3392:	c7 02 01 00 00 00    	movl   $0x1,(%rdx)
    3398:	4c 01 f2             	add    %r14,%rdx
    339b:	39 c5                	cmp    %eax,%ebp
    339d:	75 f1                	jne    3390 <blaum_roth_coding_bitmatrix+0x1c0>
    339f:	ff c7                	inc    %edi
    33a1:	41 01 ed             	add    %ebp,%r13d
    33a4:	ff ce                	dec    %esi
    33a6:	4c 03 3c 24          	add    (%rsp),%r15
    33aa:	39 fb                	cmp    %edi,%ebx
    33ac:	0f 85 1e ff ff ff    	jne    32d0 <blaum_roth_coding_bitmatrix+0x100>
    33b2:	eb b4                	jmp    3368 <blaum_roth_coding_bitmatrix+0x198>
    33b4:	45 31 c9             	xor    %r9d,%r9d
    33b7:	eb af                	jmp    3368 <blaum_roth_coding_bitmatrix+0x198>
    33b9:	45 31 c9             	xor    %r9d,%r9d
    33bc:	4c 89 c8             	mov    %r9,%rax
    33bf:	c3                   	retq   

00000000000033c0 <jerasure_print_matrix>:
static double jerasure_total_xor_bytes = 0;
static double jerasure_total_gf_bytes = 0;
static double jerasure_total_memcpy_bytes = 0;

void jerasure_print_matrix(int *m, int rows, int cols, int w)
{
    33c0:	f3 0f 1e fa          	endbr64 
    33c4:	41 57                	push   %r15
    33c6:	41 56                	push   %r14
    33c8:	41 89 d6             	mov    %edx,%r14d
    33cb:	41 55                	push   %r13
    33cd:	41 54                	push   %r12
    33cf:	55                   	push   %rbp
    33d0:	bd 0a 00 00 00       	mov    $0xa,%ebp
    33d5:	53                   	push   %rbx
    33d6:	48 83 ec 48          	sub    $0x48,%rsp
    33da:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
    33df:	89 74 24 04          	mov    %esi,0x4(%rsp)
    33e3:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    33ea:	00 00 
    33ec:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    33f1:	31 c0                	xor    %eax,%eax
  int i, j;
  int fw;
  char s[30];
  unsigned int w2;

  if (w == 32) {
    33f3:	83 f9 20             	cmp    $0x20,%ecx
    33f6:	0f 85 a1 00 00 00    	jne    349d <jerasure_print_matrix+0xdd>
    w2 = (1 << w);
    sprintf(s, "%u", w2-1);
    fw = strlen(s);
  }

  for (i = 0; i < rows; i++) {
    33fc:	8b 44 24 04          	mov    0x4(%rsp),%eax
    3400:	85 c0                	test   %eax,%eax
    3402:	7e 7a                	jle    347e <jerasure_print_matrix+0xbe>
    3404:	c7 04 24 00 00 00 00 	movl   $0x0,(%rsp)
    340b:	45 31 ff             	xor    %r15d,%r15d
  return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
    340e:	4c 8d 2d de 72 00 00 	lea    0x72de(%rip),%r13        # a6f3 <__PRETTY_FUNCTION__.5741+0x1a>
    3415:	0f 1f 00             	nopl   (%rax)
    for (j = 0; j < cols; j++) {
    3418:	45 85 f6             	test   %r14d,%r14d
    341b:	7e 49                	jle    3466 <jerasure_print_matrix+0xa6>
    341d:	48 63 14 24          	movslq (%rsp),%rdx
    3421:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
    3426:	41 8d 46 ff          	lea    -0x1(%r14),%eax
    342a:	48 01 d0             	add    %rdx,%rax
    342d:	48 8d 1c 96          	lea    (%rsi,%rdx,4),%rbx
    3431:	4c 8d 64 86 04       	lea    0x4(%rsi,%rax,4),%r12
    3436:	eb 12                	jmp    344a <jerasure_print_matrix+0x8a>
    3438:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    343f:	00 
    3440:	bf 20 00 00 00       	mov    $0x20,%edi
    3445:	e8 46 de ff ff       	callq  1290 <putchar@plt>
    344a:	8b 0b                	mov    (%rbx),%ecx
    344c:	89 ea                	mov    %ebp,%edx
    344e:	4c 89 ee             	mov    %r13,%rsi
    3451:	bf 01 00 00 00       	mov    $0x1,%edi
    3456:	31 c0                	xor    %eax,%eax
    3458:	48 83 c3 04          	add    $0x4,%rbx
    345c:	e8 cf df ff ff       	callq  1430 <__printf_chk@plt>
    3461:	4c 39 e3             	cmp    %r12,%rbx
    3464:	75 da                	jne    3440 <jerasure_print_matrix+0x80>
    3466:	bf 0a 00 00 00       	mov    $0xa,%edi
  for (i = 0; i < rows; i++) {
    346b:	41 ff c7             	inc    %r15d
    346e:	e8 1d de ff ff       	callq  1290 <putchar@plt>
    3473:	44 01 34 24          	add    %r14d,(%rsp)
    3477:	44 39 7c 24 04       	cmp    %r15d,0x4(%rsp)
    347c:	75 9a                	jne    3418 <jerasure_print_matrix+0x58>
      if (j != 0) printf(" ");
      printf("%*u", fw, m[i*cols+j]); 
    }
    printf("\n");
  }
}
    347e:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    3483:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    348a:	00 00 
    348c:	75 7d                	jne    350b <jerasure_print_matrix+0x14b>
    348e:	48 83 c4 48          	add    $0x48,%rsp
    3492:	5b                   	pop    %rbx
    3493:	5d                   	pop    %rbp
    3494:	41 5c                	pop    %r12
    3496:	41 5d                	pop    %r13
    3498:	41 5e                	pop    %r14
    349a:	41 5f                	pop    %r15
    349c:	c3                   	retq   
    w2 = (1 << w);
    349d:	b8 01 00 00 00       	mov    $0x1,%eax
  return __builtin___sprintf_chk (__s, __USE_FORTIFY_LEVEL - 1,
    34a2:	48 8d 5c 24 10       	lea    0x10(%rsp),%rbx
    34a7:	c4 e2 71 f7 c8       	shlx   %ecx,%eax,%ecx
    34ac:	ba 1e 00 00 00       	mov    $0x1e,%edx
    34b1:	44 8d 41 ff          	lea    -0x1(%rcx),%r8d
    34b5:	be 01 00 00 00       	mov    $0x1,%esi
    34ba:	48 8d 0d 2f 72 00 00 	lea    0x722f(%rip),%rcx        # a6f0 <__PRETTY_FUNCTION__.5741+0x17>
    34c1:	48 89 df             	mov    %rbx,%rdi
    34c4:	31 c0                	xor    %eax,%eax
    34c6:	e8 e5 df ff ff       	callq  14b0 <__sprintf_chk@plt>
    fw = strlen(s);
    34cb:	48 89 dd             	mov    %rbx,%rbp
    34ce:	8b 55 00             	mov    0x0(%rbp),%edx
    34d1:	48 83 c5 04          	add    $0x4,%rbp
    34d5:	8d 82 ff fe fe fe    	lea    -0x1010101(%rdx),%eax
    34db:	c4 e2 68 f2 c0       	andn   %eax,%edx,%eax
    34e0:	25 80 80 80 80       	and    $0x80808080,%eax
    34e5:	74 e7                	je     34ce <jerasure_print_matrix+0x10e>
    34e7:	89 c2                	mov    %eax,%edx
    34e9:	c1 ea 10             	shr    $0x10,%edx
    34ec:	a9 80 80 00 00       	test   $0x8080,%eax
    34f1:	0f 44 c2             	cmove  %edx,%eax
    34f4:	48 8d 55 02          	lea    0x2(%rbp),%rdx
    34f8:	48 0f 44 ea          	cmove  %rdx,%rbp
    34fc:	89 c1                	mov    %eax,%ecx
    34fe:	00 c1                	add    %al,%cl
    3500:	48 83 dd 03          	sbb    $0x3,%rbp
    3504:	29 dd                	sub    %ebx,%ebp
    3506:	e9 f1 fe ff ff       	jmpq   33fc <jerasure_print_matrix+0x3c>
}
    350b:	e8 00 de ff ff       	callq  1310 <__stack_chk_fail@plt>

0000000000003510 <jerasure_print_bitmatrix>:

void jerasure_print_bitmatrix(int *m, int rows, int cols, int w)
{
    3510:	f3 0f 1e fa          	endbr64 
    3514:	41 57                	push   %r15
    3516:	41 56                	push   %r14
    3518:	41 55                	push   %r13
    351a:	41 54                	push   %r12
    351c:	55                   	push   %rbp
    351d:	53                   	push   %rbx
    351e:	48 83 ec 28          	sub    $0x28,%rsp
    3522:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
    3527:	89 74 24 14          	mov    %esi,0x14(%rsp)
    352b:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  int i, j;

  for (i = 0; i < rows; i++) {
    352f:	85 f6                	test   %esi,%esi
    3531:	0f 8e c1 00 00 00    	jle    35f8 <jerasure_print_bitmatrix+0xe8>
    3537:	c7 44 24 10 00 00 00 	movl   $0x0,0x10(%rsp)
    353e:	00 
    353f:	44 8d 62 ff          	lea    -0x1(%rdx),%r12d
    3543:	89 cb                	mov    %ecx,%ebx
    3545:	45 31 f6             	xor    %r14d,%r14d
    3548:	49 ff c4             	inc    %r12
  return __printf_chk (__USE_FORTIFY_LEVEL - 1, __fmt, __va_arg_pack ());
    354b:	4c 8d 2d e5 6a 00 00 	lea    0x6ae5(%rip),%r13        # a037 <_IO_stdin_used+0x37>
    3552:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    if (i != 0 && i%w == 0) printf("\n");
    for (j = 0; j < cols; j++) {
    3558:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    355c:	85 c0                	test   %eax,%eax
    355e:	7e 58                	jle    35b8 <jerasure_print_bitmatrix+0xa8>
    3560:	48 63 44 24 10       	movslq 0x10(%rsp),%rax
    3565:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
      if (j != 0 && j%w == 0) printf(" ");
      printf("%d", m[i*cols+j]); 
    356a:	41 bf 01 00 00 00    	mov    $0x1,%r15d
    3570:	48 8d 2c 81          	lea    (%rcx,%rax,4),%rbp
    3574:	eb 0d                	jmp    3583 <jerasure_print_bitmatrix+0x73>
    3576:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    357d:	00 00 00 
    3580:	49 ff c7             	inc    %r15
    3583:	42 8b 54 bd fc       	mov    -0x4(%rbp,%r15,4),%edx
    3588:	4c 89 ee             	mov    %r13,%rsi
    358b:	bf 01 00 00 00       	mov    $0x1,%edi
    3590:	31 c0                	xor    %eax,%eax
    3592:	e8 99 de ff ff       	callq  1430 <__printf_chk@plt>
    for (j = 0; j < cols; j++) {
    3597:	44 89 f8             	mov    %r15d,%eax
    359a:	4d 39 fc             	cmp    %r15,%r12
    359d:	74 19                	je     35b8 <jerasure_print_bitmatrix+0xa8>
      if (j != 0 && j%w == 0) printf(" ");
    359f:	99                   	cltd   
    35a0:	f7 fb                	idiv   %ebx
    35a2:	85 d2                	test   %edx,%edx
    35a4:	75 da                	jne    3580 <jerasure_print_bitmatrix+0x70>
    35a6:	bf 20 00 00 00       	mov    $0x20,%edi
    35ab:	e8 e0 dc ff ff       	callq  1290 <putchar@plt>
    35b0:	eb ce                	jmp    3580 <jerasure_print_bitmatrix+0x70>
    35b2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    35b8:	bf 0a 00 00 00       	mov    $0xa,%edi
  for (i = 0; i < rows; i++) {
    35bd:	41 ff c6             	inc    %r14d
    35c0:	e8 cb dc ff ff       	callq  1290 <putchar@plt>
    35c5:	44 39 74 24 14       	cmp    %r14d,0x14(%rsp)
    35ca:	74 2c                	je     35f8 <jerasure_print_bitmatrix+0xe8>
    if (i != 0 && i%w == 0) printf("\n");
    35cc:	44 89 f0             	mov    %r14d,%eax
    35cf:	99                   	cltd   
    35d0:	f7 fb                	idiv   %ebx
    35d2:	85 d2                	test   %edx,%edx
    35d4:	74 12                	je     35e8 <jerasure_print_bitmatrix+0xd8>
    35d6:	8b 4c 24 0c          	mov    0xc(%rsp),%ecx
    35da:	01 4c 24 10          	add    %ecx,0x10(%rsp)
    35de:	e9 75 ff ff ff       	jmpq   3558 <jerasure_print_bitmatrix+0x48>
    35e3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    35e8:	bf 0a 00 00 00       	mov    $0xa,%edi
    35ed:	e8 9e dc ff ff       	callq  1290 <putchar@plt>
    35f2:	eb e2                	jmp    35d6 <jerasure_print_bitmatrix+0xc6>
    35f4:	0f 1f 40 00          	nopl   0x0(%rax)
    }
    printf("\n");
  }
}
    35f8:	48 83 c4 28          	add    $0x28,%rsp
    35fc:	5b                   	pop    %rbx
    35fd:	5d                   	pop    %rbp
    35fe:	41 5c                	pop    %r12
    3600:	41 5d                	pop    %r13
    3602:	41 5e                	pop    %r14
    3604:	41 5f                	pop    %r15
    3606:	c3                   	retq   
    3607:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    360e:	00 00 

0000000000003610 <jerasure_matrix_to_bitmatrix>:
  return 0;
}


int *jerasure_matrix_to_bitmatrix(int k, int m, int w, int *matrix) 
{
    3610:	f3 0f 1e fa          	endbr64 
    3614:	41 57                	push   %r15
    3616:	41 89 f7             	mov    %esi,%r15d
    3619:	41 56                	push   %r14
    361b:	41 89 d6             	mov    %edx,%r14d
    361e:	41 55                	push   %r13
    3620:	41 54                	push   %r12
    3622:	55                   	push   %rbp
    3623:	48 89 cd             	mov    %rcx,%rbp
    3626:	53                   	push   %rbx
    3627:	89 fb                	mov    %edi,%ebx
    3629:	48 83 ec 68          	sub    $0x68,%rsp
    362d:	89 7c 24 40          	mov    %edi,0x40(%rsp)
  int *bitmatrix;
  int rowelts, rowindex, colindex, elt, i, j, l, x;

  bitmatrix = talloc(int, k*m*w*w);
    3631:	0f af fe             	imul   %esi,%edi
{
    3634:	89 74 24 58          	mov    %esi,0x58(%rsp)
    3638:	48 89 4c 24 50       	mov    %rcx,0x50(%rsp)
  bitmatrix = talloc(int, k*m*w*w);
    363d:	0f af fa             	imul   %edx,%edi
    3640:	0f af fa             	imul   %edx,%edi
    3643:	48 63 ff             	movslq %edi,%rdi
    3646:	48 c1 e7 02          	shl    $0x2,%rdi
    364a:	e8 b1 dd ff ff       	callq  1400 <malloc@plt>
    364f:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  if (matrix == NULL) { return NULL; }
    3654:	48 85 ed             	test   %rbp,%rbp
    3657:	0f 84 55 01 00 00    	je     37b2 <jerasure_matrix_to_bitmatrix+0x1a2>
    365d:	48 89 c7             	mov    %rax,%rdi

  rowelts = k * w;
    3660:	89 d8                	mov    %ebx,%eax
    3662:	41 0f af c6          	imul   %r14d,%eax
  rowindex = 0;

  for (i = 0; i < m; i++) {
    3666:	45 85 ff             	test   %r15d,%r15d
    3669:	0f 8e 2f 01 00 00    	jle    379e <jerasure_matrix_to_bitmatrix+0x18e>
        }
        elt = galois_single_multiply(elt, 2, w);
      }
      colindex += w;
    }
    rowindex += rowelts * w;
    366f:	89 c1                	mov    %eax,%ecx
    3671:	41 0f af ce          	imul   %r14d,%ecx
    3675:	48 98                	cltq   
    3677:	4d 63 ee             	movslq %r14d,%r13
    367a:	89 4c 24 44          	mov    %ecx,0x44(%rsp)
    367e:	48 63 c9             	movslq %ecx,%rcx
    3681:	48 c1 e1 02          	shl    $0x2,%rcx
    3685:	48 89 4c 24 48       	mov    %rcx,0x48(%rsp)
    368a:	4c 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%r15
    3691:	00 
    3692:	4a 8d 0c ad 00 00 00 	lea    0x0(,%r13,4),%rcx
    3699:	00 
    369a:	8d 43 ff             	lea    -0x1(%rbx),%eax
    369d:	48 89 7c 24 28       	mov    %rdi,0x28(%rsp)
    36a2:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    36a7:	c7 44 24 34 00 00 00 	movl   $0x0,0x34(%rsp)
    36ae:	00 
  for (i = 0; i < m; i++) {
    36af:	c7 44 24 24 00 00 00 	movl   $0x0,0x24(%rsp)
    36b6:	00 
  rowindex = 0;
    36b7:	c7 44 24 30 00 00 00 	movl   $0x0,0x30(%rsp)
    36be:	00 
    36bf:	89 44 24 5c          	mov    %eax,0x5c(%rsp)
    for (j = 0; j < k; j++) {
    36c3:	8b 44 24 40          	mov    0x40(%rsp),%eax
    36c7:	85 c0                	test   %eax,%eax
    36c9:	0f 8e a3 00 00 00    	jle    3772 <jerasure_matrix_to_bitmatrix+0x162>
    36cf:	8b 44 24 30          	mov    0x30(%rsp),%eax
    36d3:	48 63 54 24 34       	movslq 0x34(%rsp),%rdx
    36d8:	42 8d 1c 30          	lea    (%rax,%r14,1),%ebx
    36dc:	8b 44 24 5c          	mov    0x5c(%rsp),%eax
    36e0:	48 8b 4c 24 50       	mov    0x50(%rsp),%rcx
    36e5:	48 01 d0             	add    %rdx,%rax
    36e8:	48 8d 44 81 04       	lea    0x4(%rcx,%rax,4),%rax
    36ed:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
      elt = matrix[i*k+j];
    36f2:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    36f7:	48 8d 2c 91          	lea    (%rcx,%rdx,4),%rbp
    36fb:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    3700:	45 89 f5             	mov    %r14d,%r13d
    3703:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    3708:	8b 7d 00             	mov    0x0(%rbp),%edi
      for (x = 0; x < w; x++) {
    370b:	45 85 ed             	test   %r13d,%r13d
    370e:	7e 47                	jle    3757 <jerasure_matrix_to_bitmatrix+0x147>
    3710:	41 89 dc             	mov    %ebx,%r12d
    3713:	4c 8b 74 24 08       	mov    0x8(%rsp),%r14
    3718:	45 29 ec             	sub    %r13d,%r12d
    371b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
      elt = matrix[i*k+j];
    3720:	4c 89 f6             	mov    %r14,%rsi
        for (l = 0; l < w; l++) {
    3723:	31 d2                	xor    %edx,%edx
    3725:	0f 1f 00             	nopl   (%rax)
          bitmatrix[colindex+x+l*rowelts] = ((elt & (1 << l)) ? 1 : 0);
    3728:	c4 e2 6a f7 c7       	sarx   %edx,%edi,%eax
        for (l = 0; l < w; l++) {
    372d:	ff c2                	inc    %edx
          bitmatrix[colindex+x+l*rowelts] = ((elt & (1 << l)) ? 1 : 0);
    372f:	83 e0 01             	and    $0x1,%eax
    3732:	89 06                	mov    %eax,(%rsi)
        for (l = 0; l < w; l++) {
    3734:	4c 01 fe             	add    %r15,%rsi
    3737:	41 39 d5             	cmp    %edx,%r13d
    373a:	75 ec                	jne    3728 <jerasure_matrix_to_bitmatrix+0x118>
        elt = galois_single_multiply(elt, 2, w);
    373c:	44 89 ea             	mov    %r13d,%edx
    373f:	be 02 00 00 00       	mov    $0x2,%esi
    3744:	e8 17 4f 00 00       	callq  8660 <galois_single_multiply>
    3749:	41 ff c4             	inc    %r12d
    374c:	89 c7                	mov    %eax,%edi
      for (x = 0; x < w; x++) {
    374e:	49 83 c6 04          	add    $0x4,%r14
    3752:	44 39 e3             	cmp    %r12d,%ebx
    3755:	75 c9                	jne    3720 <jerasure_matrix_to_bitmatrix+0x110>
    for (j = 0; j < k; j++) {
    3757:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    375c:	48 83 c5 04          	add    $0x4,%rbp
    3760:	48 01 4c 24 08       	add    %rcx,0x8(%rsp)
    3765:	44 01 eb             	add    %r13d,%ebx
    3768:	48 3b 6c 24 18       	cmp    0x18(%rsp),%rbp
    376d:	75 99                	jne    3708 <jerasure_matrix_to_bitmatrix+0xf8>
    376f:	45 89 ee             	mov    %r13d,%r14d
  for (i = 0; i < m; i++) {
    3772:	ff 44 24 24          	incl   0x24(%rsp)
    3776:	48 8b 5c 24 48       	mov    0x48(%rsp),%rbx
    rowindex += rowelts * w;
    377b:	8b 4c 24 44          	mov    0x44(%rsp),%ecx
    377f:	48 01 5c 24 28       	add    %rbx,0x28(%rsp)
  for (i = 0; i < m; i++) {
    3784:	8b 44 24 24          	mov    0x24(%rsp),%eax
    3788:	8b 5c 24 40          	mov    0x40(%rsp),%ebx
    rowindex += rowelts * w;
    378c:	01 4c 24 30          	add    %ecx,0x30(%rsp)
    3790:	01 5c 24 34          	add    %ebx,0x34(%rsp)
  for (i = 0; i < m; i++) {
    3794:	39 44 24 58          	cmp    %eax,0x58(%rsp)
    3798:	0f 85 25 ff ff ff    	jne    36c3 <jerasure_matrix_to_bitmatrix+0xb3>
  }
  return bitmatrix;
}
    379e:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    37a3:	48 83 c4 68          	add    $0x68,%rsp
    37a7:	5b                   	pop    %rbx
    37a8:	5d                   	pop    %rbp
    37a9:	41 5c                	pop    %r12
    37ab:	41 5d                	pop    %r13
    37ad:	41 5e                	pop    %r14
    37af:	41 5f                	pop    %r15
    37b1:	c3                   	retq   
  if (matrix == NULL) { return NULL; }
    37b2:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
    37b9:	00 00 
    37bb:	eb e1                	jmp    379e <jerasure_matrix_to_bitmatrix+0x18e>
    37bd:	0f 1f 00             	nopl   (%rax)

00000000000037c0 <jerasure_bitmatrix_dotprod>:
}

void jerasure_bitmatrix_dotprod(int k, int w, int *bitmatrix_row,
                             int *src_ids, int dest_id,
                             char **data_ptrs, char **coding_ptrs, int size, int packetsize)
{
    37c0:	f3 0f 1e fa          	endbr64 
    37c4:	41 57                	push   %r15
    37c6:	89 f0                	mov    %esi,%eax
    37c8:	41 56                	push   %r14
    37ca:	41 55                	push   %r13
    37cc:	41 54                	push   %r12
    37ce:	55                   	push   %rbp
    37cf:	53                   	push   %rbx
    37d0:	48 81 ec 88 00 00 00 	sub    $0x88,%rsp
    37d7:	44 8b ac 24 d0 00 00 	mov    0xd0(%rsp),%r13d
    37de:	00 
    37df:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
  int j, sindex, pstarted, index, x, y;
  char *dptr, *pptr, *bdptr, *bpptr;

  if (size%(w*packetsize) != 0) {
    37e4:	41 0f af c5          	imul   %r13d,%eax
{
    37e8:	48 89 54 24 30       	mov    %rdx,0x30(%rsp)
    37ed:	89 7c 24 40          	mov    %edi,0x40(%rsp)
  if (size%(w*packetsize) != 0) {
    37f1:	89 c1                	mov    %eax,%ecx
    37f3:	89 44 24 6c          	mov    %eax,0x6c(%rsp)
    37f7:	8b 84 24 c8 00 00 00 	mov    0xc8(%rsp),%eax
{
    37fe:	89 74 24 2c          	mov    %esi,0x2c(%rsp)
  if (size%(w*packetsize) != 0) {
    3802:	99                   	cltd   
    3803:	f7 f9                	idiv   %ecx
{
    3805:	4c 89 4c 24 50       	mov    %r9,0x50(%rsp)
  if (size%(w*packetsize) != 0) {
    380a:	85 d2                	test   %edx,%edx
    380c:	0f 85 d1 02 00 00    	jne    3ae3 <jerasure_bitmatrix_dotprod+0x323>
    3812:	49 63 f0             	movslq %r8d,%rsi
    fprintf(stderr, "jerasure_bitmatrix_dotprod - size%c(w*packetsize)) must = 0\n", '%');
    exit(1);
  }

  bpptr = (dest_id < k) ? data_ptrs[dest_id] : coding_ptrs[dest_id-k];
    3815:	3b 74 24 40          	cmp    0x40(%rsp),%esi
    3819:	0f 8c 1d 02 00 00    	jl     3a3c <jerasure_bitmatrix_dotprod+0x27c>
    381f:	89 f0                	mov    %esi,%eax
    3821:	48 8b bc 24 c0 00 00 	mov    0xc0(%rsp),%rdi
    3828:	00 
    3829:	2b 44 24 40          	sub    0x40(%rsp),%eax
    382d:	48 98                	cltq   
    382f:	48 8b 04 c7          	mov    (%rdi,%rax,8),%rax

  for (sindex = 0; sindex < size; sindex += (packetsize*w)) {
    3833:	8b b4 24 c8 00 00 00 	mov    0xc8(%rsp),%esi
  bpptr = (dest_id < k) ? data_ptrs[dest_id] : coding_ptrs[dest_id-k];
    383a:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
  for (sindex = 0; sindex < size; sindex += (packetsize*w)) {
    383f:	85 f6                	test   %esi,%esi
    3841:	0f 8e 12 02 00 00    	jle    3a59 <jerasure_bitmatrix_dotprod+0x299>
    3847:	48 63 44 24 6c       	movslq 0x6c(%rsp),%rax
    384c:	c7 44 24 68 00 00 00 	movl   $0x0,0x68(%rsp)
    3853:	00 
    3854:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    3859:	8b 44 24 40          	mov    0x40(%rsp),%eax
    385d:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    3864:	00 00 
    3866:	ff c8                	dec    %eax
    3868:	89 44 24 60          	mov    %eax,0x60(%rsp)
    386c:	0f af 44 24 2c       	imul   0x2c(%rsp),%eax
    3871:	45 89 ef             	mov    %r13d,%r15d
    3874:	89 44 24 64          	mov    %eax,0x64(%rsp)
    index = 0;
    for (j = 0; j < w; j++) {
    3878:	8b 44 24 2c          	mov    0x2c(%rsp),%eax
    387c:	85 c0                	test   %eax,%eax
    387e:	0f 8e 37 02 00 00    	jle    3abb <jerasure_bitmatrix_dotprod+0x2fb>
    3884:	49 63 c7             	movslq %r15d,%rax
    3887:	4c 8b 74 24 70       	mov    0x70(%rsp),%r14
    388c:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    index = 0;
    3891:	c7 44 24 5c 00 00 00 	movl   $0x0,0x5c(%rsp)
    3898:	00 
    for (j = 0; j < w; j++) {
    3899:	c7 44 24 58 00 00 00 	movl   $0x0,0x58(%rsp)
    38a0:	00 
    38a1:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    38a6:	45 89 fc             	mov    %r15d,%r12d
    38a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
      pstarted = 0;
      pptr = bpptr + sindex + j*packetsize;
      for (x = 0; x < k; x++) {
    38b0:	8b 4c 24 40          	mov    0x40(%rsp),%ecx
    38b4:	85 c9                	test   %ecx,%ecx
    38b6:	0f 8e e5 01 00 00    	jle    3aa1 <jerasure_bitmatrix_dotprod+0x2e1>
    38bc:	8b 44 24 60          	mov    0x60(%rsp),%eax
      pstarted = 0;
    38c0:	45 31 ff             	xor    %r15d,%r15d
    38c3:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
      for (x = 0; x < k; x++) {
    38c8:	8b 44 24 5c          	mov    0x5c(%rsp),%eax
    38cc:	45 89 e5             	mov    %r12d,%r13d
    38cf:	89 44 24 28          	mov    %eax,0x28(%rsp)
    38d3:	8b 44 24 2c          	mov    0x2c(%rsp),%eax
    38d7:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    38de:	00 00 
    38e0:	ff c8                	dec    %eax
    38e2:	89 44 24 44          	mov    %eax,0x44(%rsp)
    38e6:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
    38eb:	45 89 fc             	mov    %r15d,%r12d
        if (src_ids == NULL) {
    38ee:	48 83 c0 04          	add    $0x4,%rax
    38f2:	48 83 7c 24 20 00    	cmpq   $0x0,0x20(%rsp)
    38f8:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    38fd:	0f 84 22 01 00 00    	je     3a25 <jerasure_bitmatrix_dotprod+0x265>
    3903:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
          bdptr = data_ptrs[x];
        } else if (src_ids[x] < k) {
    3908:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    390d:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    3912:	48 63 14 b8          	movslq (%rax,%rdi,4),%rdx
    3916:	3b 54 24 40          	cmp    0x40(%rsp),%edx
    391a:	0f 8d 50 01 00 00    	jge    3a70 <jerasure_bitmatrix_dotprod+0x2b0>
          bdptr = data_ptrs[src_ids[x]];
    3920:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    3925:	48 8b 04 d0          	mov    (%rax,%rdx,8),%rax
    3929:	48 89 04 24          	mov    %rax,(%rsp)
        } else {
          bdptr = coding_ptrs[src_ids[x]-k];
        }
        for (y = 0; y < w; y++) {
    392d:	48 63 74 24 28       	movslq 0x28(%rsp),%rsi
    3932:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
    3937:	44 89 ac 24 d0 00 00 	mov    %r13d,0xd0(%rsp)
    393e:	00 
    393f:	8b 54 24 44          	mov    0x44(%rsp),%edx
    3943:	48 8d 2c b0          	lea    (%rax,%rsi,4),%rbp
    3947:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
    394c:	45 31 ff             	xor    %r15d,%r15d
    394f:	48 01 f2             	add    %rsi,%rdx
    3952:	45 89 e5             	mov    %r12d,%r13d
    3955:	48 8d 1c 90          	lea    (%rax,%rdx,4),%rbx
    3959:	4d 89 f4             	mov    %r14,%r12
    395c:	45 89 fe             	mov    %r15d,%r14d
    395f:	44 8b bc 24 d0 00 00 	mov    0xd0(%rsp),%r15d
    3966:	00 
    3967:	eb 3f                	jmp    39a8 <jerasure_bitmatrix_dotprod+0x1e8>
    3969:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  return __builtin___memcpy_chk (__dest, __src, __len, __bos0 (__dest));
    3970:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
    3975:	4c 89 e7             	mov    %r12,%rdi
    3978:	e8 43 da ff ff       	callq  13c0 <memcpy@plt>
          if (bitmatrix_row[index]) {
            dptr = bdptr + sindex + y*packetsize;
            if (!pstarted) {
              memcpy(pptr, dptr, packetsize);
              jerasure_total_memcpy_bytes += packetsize;
    397d:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
    3981:	c4 c1 73 2a c7       	vcvtsi2sd %r15d,%xmm1,%xmm0
              pstarted = 1;
    3986:	41 bd 01 00 00 00    	mov    $0x1,%r13d
              jerasure_total_memcpy_bytes += packetsize;
    398c:	c5 fb 58 05 bc d7 00 	vaddsd 0xd7bc(%rip),%xmm0,%xmm0        # 11150 <jerasure_total_memcpy_bytes>
    3993:	00 
    3994:	c5 fb 11 05 b4 d7 00 	vmovsd %xmm0,0xd7b4(%rip)        # 11150 <jerasure_total_memcpy_bytes>
    399b:	00 
        for (y = 0; y < w; y++) {
    399c:	48 83 c5 04          	add    $0x4,%rbp
    39a0:	45 01 fe             	add    %r15d,%r14d
    39a3:	48 39 dd             	cmp    %rbx,%rbp
    39a6:	74 4b                	je     39f3 <jerasure_bitmatrix_dotprod+0x233>
          if (bitmatrix_row[index]) {
    39a8:	8b 55 00             	mov    0x0(%rbp),%edx
    39ab:	85 d2                	test   %edx,%edx
    39ad:	74 ed                	je     399c <jerasure_bitmatrix_dotprod+0x1dc>
            dptr = bdptr + sindex + y*packetsize;
    39af:	49 63 f6             	movslq %r14d,%rsi
    39b2:	48 03 74 24 08       	add    0x8(%rsp),%rsi
    39b7:	48 03 34 24          	add    (%rsp),%rsi
            if (!pstarted) {
    39bb:	45 85 ed             	test   %r13d,%r13d
    39be:	74 b0                	je     3970 <jerasure_bitmatrix_dotprod+0x1b0>
            } else {
              galois_region_xor(pptr, dptr, pptr, packetsize);
    39c0:	44 89 f9             	mov    %r15d,%ecx
    39c3:	4c 89 e2             	mov    %r12,%rdx
    39c6:	4c 89 e7             	mov    %r12,%rdi
    39c9:	e8 52 47 00 00       	callq  8120 <galois_region_xor>
              jerasure_total_xor_bytes += packetsize;
    39ce:	c5 e9 57 d2          	vxorpd %xmm2,%xmm2,%xmm2
    39d2:	c4 c1 6b 2a c7       	vcvtsi2sd %r15d,%xmm2,%xmm0
    39d7:	48 83 c5 04          	add    $0x4,%rbp
    39db:	45 01 fe             	add    %r15d,%r14d
    39de:	c5 fb 58 05 7a d7 00 	vaddsd 0xd77a(%rip),%xmm0,%xmm0        # 11160 <jerasure_total_xor_bytes>
    39e5:	00 
    39e6:	c5 fb 11 05 72 d7 00 	vmovsd %xmm0,0xd772(%rip)        # 11160 <jerasure_total_xor_bytes>
    39ed:	00 
        for (y = 0; y < w; y++) {
    39ee:	48 39 dd             	cmp    %rbx,%rbp
    39f1:	75 b5                	jne    39a8 <jerasure_bitmatrix_dotprod+0x1e8>
      for (x = 0; x < k; x++) {
    39f3:	8b 4c 24 2c          	mov    0x2c(%rsp),%ecx
    39f7:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    39fc:	4d 89 e6             	mov    %r12,%r14
    39ff:	01 4c 24 28          	add    %ecx,0x28(%rsp)
    3a03:	45 89 ec             	mov    %r13d,%r12d
    3a06:	48 8d 50 01          	lea    0x1(%rax),%rdx
    3a0a:	45 89 fd             	mov    %r15d,%r13d
    3a0d:	48 3b 44 24 38       	cmp    0x38(%rsp),%rax
    3a12:	74 7c                	je     3a90 <jerasure_bitmatrix_dotprod+0x2d0>
        if (src_ids == NULL) {
    3a14:	48 83 7c 24 20 00    	cmpq   $0x0,0x20(%rsp)
    3a1a:	48 89 54 24 10       	mov    %rdx,0x10(%rsp)
    3a1f:	0f 85 e3 fe ff ff    	jne    3908 <jerasure_bitmatrix_dotprod+0x148>
          bdptr = data_ptrs[x];
    3a25:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    3a2a:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    3a2f:	48 8b 04 f8          	mov    (%rax,%rdi,8),%rax
    3a33:	48 89 04 24          	mov    %rax,(%rsp)
    3a37:	e9 f1 fe ff ff       	jmpq   392d <jerasure_bitmatrix_dotprod+0x16d>
  bpptr = (dest_id < k) ? data_ptrs[dest_id] : coding_ptrs[dest_id-k];
    3a3c:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    3a41:	48 8b 04 f0          	mov    (%rax,%rsi,8),%rax
  for (sindex = 0; sindex < size; sindex += (packetsize*w)) {
    3a45:	8b b4 24 c8 00 00 00 	mov    0xc8(%rsp),%esi
  bpptr = (dest_id < k) ? data_ptrs[dest_id] : coding_ptrs[dest_id-k];
    3a4c:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
  for (sindex = 0; sindex < size; sindex += (packetsize*w)) {
    3a51:	85 f6                	test   %esi,%esi
    3a53:	0f 8f ee fd ff ff    	jg     3847 <jerasure_bitmatrix_dotprod+0x87>
          index++;
        }
      }
    }
  }
}
    3a59:	48 81 c4 88 00 00 00 	add    $0x88,%rsp
    3a60:	5b                   	pop    %rbx
    3a61:	5d                   	pop    %rbp
    3a62:	41 5c                	pop    %r12
    3a64:	41 5d                	pop    %r13
    3a66:	41 5e                	pop    %r14
    3a68:	41 5f                	pop    %r15
    3a6a:	c3                   	retq   
    3a6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
          bdptr = coding_ptrs[src_ids[x]-k];
    3a70:	48 8b 84 24 c0 00 00 	mov    0xc0(%rsp),%rax
    3a77:	00 
    3a78:	2b 54 24 40          	sub    0x40(%rsp),%edx
    3a7c:	48 63 d2             	movslq %edx,%rdx
    3a7f:	48 8b 04 d0          	mov    (%rax,%rdx,8),%rax
    3a83:	48 89 04 24          	mov    %rax,(%rsp)
    3a87:	e9 a1 fe ff ff       	jmpq   392d <jerasure_bitmatrix_dotprod+0x16d>
    3a8c:	0f 1f 40 00          	nopl   0x0(%rax)
    3a90:	8b 44 24 5c          	mov    0x5c(%rsp),%eax
    3a94:	45 89 fc             	mov    %r15d,%r12d
    3a97:	01 c8                	add    %ecx,%eax
    3a99:	03 44 24 64          	add    0x64(%rsp),%eax
    3a9d:	89 44 24 5c          	mov    %eax,0x5c(%rsp)
    for (j = 0; j < w; j++) {
    3aa1:	ff 44 24 58          	incl   0x58(%rsp)
    3aa5:	4c 03 74 24 18       	add    0x18(%rsp),%r14
    3aaa:	8b 44 24 58          	mov    0x58(%rsp),%eax
    3aae:	39 44 24 2c          	cmp    %eax,0x2c(%rsp)
    3ab2:	0f 85 f8 fd ff ff    	jne    38b0 <jerasure_bitmatrix_dotprod+0xf0>
    3ab8:	45 89 e7             	mov    %r12d,%r15d
  for (sindex = 0; sindex < size; sindex += (packetsize*w)) {
    3abb:	8b 7c 24 6c          	mov    0x6c(%rsp),%edi
    3abf:	48 8b 4c 24 78       	mov    0x78(%rsp),%rcx
    3ac4:	01 7c 24 68          	add    %edi,0x68(%rsp)
    3ac8:	48 01 4c 24 08       	add    %rcx,0x8(%rsp)
    3acd:	8b 44 24 68          	mov    0x68(%rsp),%eax
    3ad1:	39 84 24 c8 00 00 00 	cmp    %eax,0xc8(%rsp)
    3ad8:	0f 8f 9a fd ff ff    	jg     3878 <jerasure_bitmatrix_dotprod+0xb8>
    3ade:	e9 76 ff ff ff       	jmpq   3a59 <jerasure_bitmatrix_dotprod+0x299>
  return __fprintf_chk (__stream, __USE_FORTIFY_LEVEL - 1, __fmt,
    3ae3:	48 8b 3d 56 d6 00 00 	mov    0xd656(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    3aea:	b9 25 00 00 00       	mov    $0x25,%ecx
    3aef:	48 8d 15 02 6c 00 00 	lea    0x6c02(%rip),%rdx        # a6f8 <__PRETTY_FUNCTION__.5741+0x1f>
    3af6:	be 01 00 00 00       	mov    $0x1,%esi
    3afb:	31 c0                	xor    %eax,%eax
    3afd:	e8 7e d9 ff ff       	callq  1480 <__fprintf_chk@plt>
    exit(1);
    3b02:	bf 01 00 00 00       	mov    $0x1,%edi
    3b07:	e8 54 d9 ff ff       	callq  1460 <exit@plt>
    3b0c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000003b10 <jerasure_do_parity>:

void jerasure_do_parity(int k, char **data_ptrs, char *parity_ptr, int size) 
{
    3b10:	f3 0f 1e fa          	endbr64 
    3b14:	41 56                	push   %r14
    3b16:	41 89 fe             	mov    %edi,%r14d
    3b19:	41 55                	push   %r13
    3b1b:	49 89 f5             	mov    %rsi,%r13
    3b1e:	41 54                	push   %r12
    3b20:	55                   	push   %rbp
    3b21:	48 89 d5             	mov    %rdx,%rbp
    3b24:	48 89 ef             	mov    %rbp,%rdi
    3b27:	53                   	push   %rbx
    3b28:	48 63 d1             	movslq %ecx,%rdx
    3b2b:	49 89 d4             	mov    %rdx,%r12
    3b2e:	48 83 ec 10          	sub    $0x10,%rsp
    3b32:	48 8b 36             	mov    (%rsi),%rsi
    3b35:	e8 86 d8 ff ff       	callq  13c0 <memcpy@plt>
  int i;

  memcpy(parity_ptr, data_ptrs[0], size);
  jerasure_total_memcpy_bytes += size;
    3b3a:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    3b3e:	c4 c1 7b 2a c4       	vcvtsi2sd %r12d,%xmm0,%xmm0
    3b43:	c5 fb 11 44 24 08    	vmovsd %xmm0,0x8(%rsp)
    3b49:	c5 fb 58 05 ff d5 00 	vaddsd 0xd5ff(%rip),%xmm0,%xmm0        # 11150 <jerasure_total_memcpy_bytes>
    3b50:	00 
    3b51:	c5 fb 11 05 f7 d5 00 	vmovsd %xmm0,0xd5f7(%rip)        # 11150 <jerasure_total_memcpy_bytes>
    3b58:	00 
  
  for (i = 1; i < k; i++) {
    3b59:	41 83 fe 01          	cmp    $0x1,%r14d
    3b5d:	7e 41                	jle    3ba0 <jerasure_do_parity+0x90>
    3b5f:	41 8d 46 fe          	lea    -0x2(%r14),%eax
    3b63:	49 8d 5d 08          	lea    0x8(%r13),%rbx
    3b67:	4d 8d 6c c5 10       	lea    0x10(%r13,%rax,8),%r13
    3b6c:	0f 1f 40 00          	nopl   0x0(%rax)
    galois_region_xor(data_ptrs[i], parity_ptr, parity_ptr, size);
    3b70:	48 8b 3b             	mov    (%rbx),%rdi
    3b73:	44 89 e1             	mov    %r12d,%ecx
    3b76:	48 89 ea             	mov    %rbp,%rdx
    3b79:	48 89 ee             	mov    %rbp,%rsi
    3b7c:	e8 9f 45 00 00       	callq  8120 <galois_region_xor>
    jerasure_total_xor_bytes += size;
    3b81:	c5 fb 10 4c 24 08    	vmovsd 0x8(%rsp),%xmm1
    3b87:	48 83 c3 08          	add    $0x8,%rbx
    3b8b:	c5 f3 58 05 cd d5 00 	vaddsd 0xd5cd(%rip),%xmm1,%xmm0        # 11160 <jerasure_total_xor_bytes>
    3b92:	00 
    3b93:	c5 fb 11 05 c5 d5 00 	vmovsd %xmm0,0xd5c5(%rip)        # 11160 <jerasure_total_xor_bytes>
    3b9a:	00 
  for (i = 1; i < k; i++) {
    3b9b:	4c 39 eb             	cmp    %r13,%rbx
    3b9e:	75 d0                	jne    3b70 <jerasure_do_parity+0x60>
  }
}
    3ba0:	48 83 c4 10          	add    $0x10,%rsp
    3ba4:	5b                   	pop    %rbx
    3ba5:	5d                   	pop    %rbp
    3ba6:	41 5c                	pop    %r12
    3ba8:	41 5d                	pop    %r13
    3baa:	41 5e                	pop    %r14
    3bac:	c3                   	retq   
    3bad:	0f 1f 00             	nopl   (%rax)

0000000000003bb0 <jerasure_invert_matrix>:

int jerasure_invert_matrix(int *mat, int *inv, int rows, int w)
{
    3bb0:	f3 0f 1e fa          	endbr64 
    3bb4:	41 57                	push   %r15
    3bb6:	41 56                	push   %r14
    3bb8:	41 55                	push   %r13
    3bba:	41 54                	push   %r12
    3bbc:	55                   	push   %rbp
    3bbd:	48 89 f5             	mov    %rsi,%rbp
  int row_start, tmp, inverse;
 
  cols = rows;

  k = 0;
  for (i = 0; i < rows; i++) {
    3bc0:	8d 72 ff             	lea    -0x1(%rdx),%esi
{
    3bc3:	53                   	push   %rbx
    3bc4:	89 cb                	mov    %ecx,%ebx
    3bc6:	48 81 ec a8 00 00 00 	sub    $0xa8,%rsp
    3bcd:	89 54 24 30          	mov    %edx,0x30(%rsp)
  for (i = 0; i < rows; i++) {
    3bd1:	c7 44 24 38 00 00 00 	movl   $0x0,0x38(%rsp)
    3bd8:	00 
    3bd9:	89 b4 24 9c 00 00 00 	mov    %esi,0x9c(%rsp)
    3be0:	85 d2                	test   %edx,%edx
    3be2:	0f 8e af 04 00 00    	jle    4097 <jerasure_invert_matrix+0x4e7>
    3be8:	44 8b 4c 24 30       	mov    0x30(%rsp),%r9d
    3bed:	49 89 fe             	mov    %rdi,%r14
    3bf0:	31 c9                	xor    %ecx,%ecx
    3bf2:	31 ff                	xor    %edi,%edi
    3bf4:	0f 1f 40 00          	nopl   0x0(%rax)
    for (j = 0; j < cols; j++) {
    3bf8:	48 63 c7             	movslq %edi,%rax
    3bfb:	4c 8d 44 85 00       	lea    0x0(%rbp,%rax,4),%r8
{
    3c00:	31 c0                	xor    %eax,%eax
    3c02:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
      inv[k] = (i == j) ? 1 : 0;
    3c08:	31 d2                	xor    %edx,%edx
    3c0a:	39 c1                	cmp    %eax,%ecx
    3c0c:	0f 94 c2             	sete   %dl
    3c0f:	41 89 14 80          	mov    %edx,(%r8,%rax,4)
    for (j = 0; j < cols; j++) {
    3c13:	48 89 c2             	mov    %rax,%rdx
    3c16:	48 ff c0             	inc    %rax
    3c19:	48 39 f2             	cmp    %rsi,%rdx
    3c1c:	75 ea                	jne    3c08 <jerasure_invert_matrix+0x58>
  for (i = 0; i < rows; i++) {
    3c1e:	8d 41 01             	lea    0x1(%rcx),%eax
    3c21:	44 01 cf             	add    %r9d,%edi
    3c24:	41 39 c1             	cmp    %eax,%r9d
    3c27:	74 04                	je     3c2d <jerasure_invert_matrix+0x7d>
    3c29:	89 c1                	mov    %eax,%ecx
    3c2b:	eb cb                	jmp    3bf8 <jerasure_invert_matrix+0x48>
    3c2d:	89 84 24 98 00 00 00 	mov    %eax,0x98(%rsp)
    3c34:	48 63 44 24 30       	movslq 0x30(%rsp),%rax
    3c39:	89 4c 24 38          	mov    %ecx,0x38(%rsp)
    3c3d:	48 8d 3c 85 04 00 00 	lea    0x4(,%rax,4),%rdi
    3c44:	00 
    3c45:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    3c4a:	48 c1 e0 02          	shl    $0x2,%rax
    3c4e:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    3c53:	89 c8                	mov    %ecx,%eax
    3c55:	49 8d 4e 04          	lea    0x4(%r14),%rcx
    3c59:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    3c5e:	48 89 8c 24 88 00 00 	mov    %rcx,0x88(%rsp)
    3c65:	00 
    3c66:	48 8d 0c 81          	lea    (%rcx,%rax,4),%rcx
    3c6a:	48 ff c0             	inc    %rax
    3c6d:	48 89 bc 24 90 00 00 	mov    %rdi,0x90(%rsp)
    3c74:	00 
    3c75:	48 89 8c 24 80 00 00 	mov    %rcx,0x80(%rsp)
    3c7c:	00 
    3c7d:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
    3c82:	4c 89 74 24 60       	mov    %r14,0x60(%rsp)
    3c87:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
    3c8e:	00 00 
    3c90:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
    3c97:	00 00 
    3c99:	c7 44 24 6c 00 00 00 	movl   $0x0,0x6c(%rsp)
    3ca0:	00 
    3ca1:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
    3ca8:	00 00 
    3caa:	49 89 ef             	mov    %rbp,%r15
    3cad:	0f 1f 00             	nopl   (%rax)
    row_start = cols*i;

    /* Swap rows if we ave a zero i,i element.  If we can't swap, then the 
       matrix was not invertible  */

    if (mat[row_start+i] == 0) { 
    3cb0:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    3cb5:	8b 30                	mov    (%rax),%esi
    3cb7:	8b 44 24 48          	mov    0x48(%rsp),%eax
    3cbb:	ff c0                	inc    %eax
    3cbd:	89 44 24 18          	mov    %eax,0x18(%rsp)
    3cc1:	85 f6                	test   %esi,%esi
    3cc3:	0f 84 87 01 00 00    	je     3e50 <jerasure_invert_matrix+0x2a0>
      }
    }
 
    /* Multiply the row by 1/element i,i  */
    tmp = mat[row_start+i];
    if (tmp != 1) {
    3cc9:	83 fe 01             	cmp    $0x1,%esi
    3ccc:	0f 85 27 02 00 00    	jne    3ef9 <jerasure_invert_matrix+0x349>
      }
    }

    /* Now for each j>i, add A_ji*Ai to Aj  */
    k = row_start+i;
    for (j = i+1; j != cols; j++) {
    3cd2:	48 8b 4c 24 58       	mov    0x58(%rsp),%rcx
    3cd7:	48 39 4c 24 48       	cmp    %rcx,0x48(%rsp)
    3cdc:	0f 84 8a 02 00 00    	je     3f6c <jerasure_invert_matrix+0x3bc>
    3ce2:	8b 4c 24 30          	mov    0x30(%rsp),%ecx
    3ce6:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    3ceb:	01 4c 24 6c          	add    %ecx,0x6c(%rsp)
    3cef:	48 03 44 24 28       	add    0x28(%rsp),%rax
    3cf4:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    3cf9:	8b 44 24 6c          	mov    0x6c(%rsp),%eax
    3cfd:	89 44 24 20          	mov    %eax,0x20(%rsp)
    3d01:	4c 89 f8             	mov    %r15,%rax
    3d04:	4d 89 f7             	mov    %r14,%r15
    3d07:	49 89 c6             	mov    %rax,%r14
    3d0a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
      k += cols;
      if (mat[k] != 0) {
    3d10:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    3d15:	8b 28                	mov    (%rax),%ebp
    3d17:	85 ed                	test   %ebp,%ebp
    3d19:	74 5e                	je     3d79 <jerasure_invert_matrix+0x1c9>
        if (mat[k] == 1) {
    3d1b:	83 fd 01             	cmp    $0x1,%ebp
    3d1e:	0f 84 84 00 00 00    	je     3da8 <jerasure_invert_matrix+0x1f8>
    3d24:	48 63 44 24 20       	movslq 0x20(%rsp),%rax
    3d29:	4c 8b 64 24 50       	mov    0x50(%rsp),%r12
    3d2e:	4c 8d 2c 85 00 00 00 	lea    0x0(,%rax,4),%r13
    3d35:	00 
    3d36:	48 03 44 24 70       	add    0x70(%rsp),%rax
    3d3b:	48 c1 e0 02          	shl    $0x2,%rax
    3d3f:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    3d44:	0f 1f 40 00          	nopl   0x0(%rax)
          }
        } else {
          tmp = mat[k];
          rs2 = cols*j;
          for (x = 0; x < cols; x++) {
            mat[rs2+x] ^= galois_single_multiply(tmp, mat[row_start+x], w);
    3d48:	43 8b 34 27          	mov    (%r15,%r12,1),%esi
    3d4c:	89 da                	mov    %ebx,%edx
    3d4e:	89 ef                	mov    %ebp,%edi
    3d50:	e8 0b 49 00 00       	callq  8660 <galois_single_multiply>
    3d55:	43 31 04 2f          	xor    %eax,(%r15,%r13,1)
            inv[rs2+x] ^= galois_single_multiply(tmp, inv[row_start+x], w);
    3d59:	89 da                	mov    %ebx,%edx
    3d5b:	89 ef                	mov    %ebp,%edi
    3d5d:	43 8b 34 26          	mov    (%r14,%r12,1),%esi
    3d61:	49 83 c4 04          	add    $0x4,%r12
    3d65:	e8 f6 48 00 00       	callq  8660 <galois_single_multiply>
    3d6a:	43 31 04 2e          	xor    %eax,(%r14,%r13,1)
          for (x = 0; x < cols; x++) {
    3d6e:	49 83 c5 04          	add    $0x4,%r13
    3d72:	4c 3b 6c 24 08       	cmp    0x8(%rsp),%r13
    3d77:	75 cf                	jne    3d48 <jerasure_invert_matrix+0x198>
    for (j = i+1; j != cols; j++) {
    3d79:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    3d7e:	8b 4c 24 18          	mov    0x18(%rsp),%ecx
    3d82:	48 01 54 24 10       	add    %rdx,0x10(%rsp)
    3d87:	8b 54 24 30          	mov    0x30(%rsp),%edx
    3d8b:	8d 41 01             	lea    0x1(%rcx),%eax
    3d8e:	01 54 24 20          	add    %edx,0x20(%rsp)
    3d92:	3b 4c 24 38          	cmp    0x38(%rsp),%ecx
    3d96:	74 68                	je     3e00 <jerasure_invert_matrix+0x250>
    3d98:	89 44 24 18          	mov    %eax,0x18(%rsp)
    3d9c:	e9 6f ff ff ff       	jmpq   3d10 <jerasure_invert_matrix+0x160>
    3da1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3da8:	48 63 4c 24 20       	movslq 0x20(%rsp),%rcx
    3dad:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    3db2:	48 8d 14 8d 00 00 00 	lea    0x0(,%rcx,4),%rdx
    3db9:	00 
    3dba:	48 8d 34 39          	lea    (%rcx,%rdi,1),%rsi
    3dbe:	48 8b bc 24 88 00 00 	mov    0x88(%rsp),%rdi
    3dc5:	00 
    3dc6:	49 8d 04 17          	lea    (%r15,%rdx,1),%rax
    3dca:	48 8d 3c b7          	lea    (%rdi,%rsi,4),%rdi
    3dce:	48 8b 74 24 78       	mov    0x78(%rsp),%rsi
    3dd3:	4c 01 f2             	add    %r14,%rdx
    3dd6:	48 29 ce             	sub    %rcx,%rsi
    3dd9:	48 89 f1             	mov    %rsi,%rcx
    3ddc:	0f 1f 40 00          	nopl   0x0(%rax)
            mat[rs2+x] ^= mat[row_start+x];
    3de0:	8b 34 88             	mov    (%rax,%rcx,4),%esi
    3de3:	31 30                	xor    %esi,(%rax)
            inv[rs2+x] ^= inv[row_start+x];
    3de5:	48 83 c0 04          	add    $0x4,%rax
    3de9:	8b 34 8a             	mov    (%rdx,%rcx,4),%esi
    3dec:	31 32                	xor    %esi,(%rdx)
          for (x = 0; x < cols; x++) {
    3dee:	48 83 c2 04          	add    $0x4,%rdx
    3df2:	48 39 f8             	cmp    %rdi,%rax
    3df5:	75 e9                	jne    3de0 <jerasure_invert_matrix+0x230>
    3df7:	eb 80                	jmp    3d79 <jerasure_invert_matrix+0x1c9>
    3df9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3e00:	48 8b 94 24 90 00 00 	mov    0x90(%rsp),%rdx
    3e07:	00 
    3e08:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    3e0d:	48 01 54 24 60       	add    %rdx,0x60(%rsp)
    3e12:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
    3e17:	4c 89 f0             	mov    %r14,%rax
    3e1a:	48 01 54 24 78       	add    %rdx,0x78(%rsp)
    3e1f:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    3e24:	4d 89 fe             	mov    %r15,%r14
    3e27:	48 01 54 24 50       	add    %rdx,0x50(%rsp)
    3e2c:	49 89 c7             	mov    %rax,%r15
  for (i = 0; i < cols; i++) {
    3e2f:	48 01 94 24 80 00 00 	add    %rdx,0x80(%rsp)
    3e36:	00 
    3e37:	48 8d 41 01          	lea    0x1(%rcx),%rax
    3e3b:	48 3b 4c 24 58       	cmp    0x58(%rsp),%rcx
    3e40:	0f 84 26 01 00 00    	je     3f6c <jerasure_invert_matrix+0x3bc>
    3e46:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    3e4b:	e9 60 fe ff ff       	jmpq   3cb0 <jerasure_invert_matrix+0x100>
      for (j = i+1; j < rows && mat[cols*j+i] == 0; j++) ;
    3e50:	89 c1                	mov    %eax,%ecx
    3e52:	39 84 24 98 00 00 00 	cmp    %eax,0x98(%rsp)
    3e59:	0f 8e 3c 02 00 00    	jle    409b <jerasure_invert_matrix+0x4eb>
    3e5f:	8b 44 24 6c          	mov    0x6c(%rsp),%eax
    3e63:	8b 74 24 38          	mov    0x38(%rsp),%esi
    3e67:	03 44 24 30          	add    0x30(%rsp),%eax
    3e6b:	48 98                	cltq   
    3e6d:	48 03 44 24 48       	add    0x48(%rsp),%rax
    3e72:	49 8d 14 86          	lea    (%r14,%rax,4),%rdx
    3e76:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    3e7b:	89 c8                	mov    %ecx,%eax
    3e7d:	eb 11                	jmp    3e90 <jerasure_invert_matrix+0x2e0>
    3e7f:	90                   	nop
    3e80:	8d 48 01             	lea    0x1(%rax),%ecx
    3e83:	48 01 fa             	add    %rdi,%rdx
    3e86:	39 f0                	cmp    %esi,%eax
    3e88:	0f 84 be 01 00 00    	je     404c <jerasure_invert_matrix+0x49c>
    3e8e:	89 c8                	mov    %ecx,%eax
    3e90:	8b 0a                	mov    (%rdx),%ecx
    3e92:	85 c9                	test   %ecx,%ecx
    3e94:	74 ea                	je     3e80 <jerasure_invert_matrix+0x2d0>
      if (j == rows) return -1;
    3e96:	8b 8c 24 98 00 00 00 	mov    0x98(%rsp),%ecx
    3e9d:	39 c1                	cmp    %eax,%ecx
    3e9f:	0f 84 a7 01 00 00    	je     404c <jerasure_invert_matrix+0x49c>
      rs2 = j*cols;
    3ea5:	0f af c1             	imul   %ecx,%eax
      for (k = 0; k < cols; k++) {
    3ea8:	48 8b 4c 24 50       	mov    0x50(%rsp),%rcx
    3ead:	4c 8b 84 24 80 00 00 	mov    0x80(%rsp),%r8
    3eb4:	00 
    3eb5:	48 98                	cltq   
    3eb7:	49 8d 14 0e          	lea    (%r14,%rcx,1),%rdx
    3ebb:	48 2b 44 24 78       	sub    0x78(%rsp),%rax
    3ec0:	4c 01 f9             	add    %r15,%rcx
    3ec3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
        tmp = mat[row_start+k];
    3ec8:	8b 32                	mov    (%rdx),%esi
        mat[row_start+k] = mat[rs2+k];
    3eca:	8b 3c 82             	mov    (%rdx,%rax,4),%edi
    3ecd:	89 3a                	mov    %edi,(%rdx)
        mat[rs2+k] = tmp;
    3ecf:	89 34 82             	mov    %esi,(%rdx,%rax,4)
        tmp = inv[row_start+k];
    3ed2:	48 83 c2 04          	add    $0x4,%rdx
    3ed6:	8b 31                	mov    (%rcx),%esi
        inv[row_start+k] = inv[rs2+k];
    3ed8:	8b 3c 81             	mov    (%rcx,%rax,4),%edi
    3edb:	89 39                	mov    %edi,(%rcx)
        inv[rs2+k] = tmp;
    3edd:	89 34 81             	mov    %esi,(%rcx,%rax,4)
      for (k = 0; k < cols; k++) {
    3ee0:	48 83 c1 04          	add    $0x4,%rcx
    3ee4:	4c 39 c2             	cmp    %r8,%rdx
    3ee7:	75 df                	jne    3ec8 <jerasure_invert_matrix+0x318>
    3ee9:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    3eee:	8b 30                	mov    (%rax),%esi
    if (tmp != 1) {
    3ef0:	83 fe 01             	cmp    $0x1,%esi
    3ef3:	0f 84 d9 fd ff ff    	je     3cd2 <jerasure_invert_matrix+0x122>
      inverse = galois_single_divide(1, tmp, w);
    3ef9:	89 da                	mov    %ebx,%edx
    3efb:	bf 01 00 00 00       	mov    $0x1,%edi
    3f00:	e8 7b 47 00 00       	callq  8680 <galois_single_divide>
    3f05:	41 89 c5             	mov    %eax,%r13d
      for (j = 0; j < cols; j++) { 
    3f08:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    3f0d:	4c 89 74 24 08       	mov    %r14,0x8(%rsp)
    3f12:	49 8d 2c 06          	lea    (%r14,%rax,1),%rbp
    3f16:	49 89 ee             	mov    %rbp,%r14
    3f19:	48 8b ac 24 80 00 00 	mov    0x80(%rsp),%rbp
    3f20:	00 
    3f21:	4d 8d 24 07          	lea    (%r15,%rax,1),%r12
    3f25:	0f 1f 00             	nopl   (%rax)
        mat[row_start+j] = galois_single_multiply(mat[row_start+j], inverse, w);
    3f28:	41 8b 3e             	mov    (%r14),%edi
    3f2b:	89 da                	mov    %ebx,%edx
    3f2d:	44 89 ee             	mov    %r13d,%esi
    3f30:	e8 2b 47 00 00       	callq  8660 <galois_single_multiply>
    3f35:	41 89 06             	mov    %eax,(%r14)
        inv[row_start+j] = galois_single_multiply(inv[row_start+j], inverse, w);
    3f38:	89 da                	mov    %ebx,%edx
    3f3a:	44 89 ee             	mov    %r13d,%esi
    3f3d:	41 8b 3c 24          	mov    (%r12),%edi
    3f41:	49 83 c6 04          	add    $0x4,%r14
    3f45:	e8 16 47 00 00       	callq  8660 <galois_single_multiply>
    3f4a:	41 89 04 24          	mov    %eax,(%r12)
      for (j = 0; j < cols; j++) { 
    3f4e:	49 83 c4 04          	add    $0x4,%r12
    3f52:	49 39 ee             	cmp    %rbp,%r14
    3f55:	75 d1                	jne    3f28 <jerasure_invert_matrix+0x378>
    for (j = i+1; j != cols; j++) {
    3f57:	48 8b 4c 24 58       	mov    0x58(%rsp),%rcx
    3f5c:	4c 8b 74 24 08       	mov    0x8(%rsp),%r14
    3f61:	48 39 4c 24 48       	cmp    %rcx,0x48(%rsp)
    3f66:	0f 85 76 fd ff ff    	jne    3ce2 <jerasure_invert_matrix+0x132>
    }
  }

  /* Now the matrix is upper triangular.  Start at the top and multiply down  */

  for (i = rows-1; i >= 0; i--) {
    3f6c:	48 63 44 24 38       	movslq 0x38(%rsp),%rax
    3f71:	8b 4c 24 30          	mov    0x30(%rsp),%ecx
    3f75:	89 44 24 10          	mov    %eax,0x10(%rsp)
    3f79:	48 89 c7             	mov    %rax,%rdi
    3f7c:	49 8d 04 86          	lea    (%r14,%rax,4),%rax
    3f80:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    3f85:	89 c8                	mov    %ecx,%eax
    3f87:	0f af cf             	imul   %edi,%ecx
    3f8a:	f7 d8                	neg    %eax
    3f8c:	48 98                	cltq   
    3f8e:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    3f93:	48 63 c1             	movslq %ecx,%rax
    3f96:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    3f9b:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    3fa0:	48 c1 e0 02          	shl    $0x2,%rax
    3fa4:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    3fa9:	8b 84 24 9c 00 00 00 	mov    0x9c(%rsp),%eax
    3fb0:	49 8d 4c 87 04       	lea    0x4(%r15,%rax,4),%rcx
    3fb5:	48 f7 d0             	not    %rax
    3fb8:	48 c1 e0 02          	shl    $0x2,%rax
    3fbc:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    row_start = i*cols;
    for (j = 0; j < i; j++) {
    3fc1:	8b 44 24 10          	mov    0x10(%rsp),%eax
    3fc5:	48 89 4c 24 48       	mov    %rcx,0x48(%rsp)
    3fca:	85 c0                	test   %eax,%eax
    3fcc:	0f 8e bb 00 00 00    	jle    408d <jerasure_invert_matrix+0x4dd>
    3fd2:	c7 44 24 08 00 00 00 	movl   $0x0,0x8(%rsp)
    3fd9:	00 
    3fda:	48 8b 6c 24 48       	mov    0x48(%rsp),%rbp
    3fdf:	4c 8b 7c 24 28       	mov    0x28(%rsp),%r15
    3fe4:	4c 8b 64 24 38       	mov    0x38(%rsp),%r12
    3fe9:	eb 23                	jmp    400e <jerasure_invert_matrix+0x45e>
    3feb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    3ff0:	ff 44 24 08          	incl   0x8(%rsp)
    3ff4:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
    3ff9:	4c 2b 7c 24 40       	sub    0x40(%rsp),%r15
    3ffe:	49 01 cc             	add    %rcx,%r12
    4001:	8b 44 24 08          	mov    0x8(%rsp),%eax
    4005:	48 01 cd             	add    %rcx,%rbp
    4008:	3b 44 24 10          	cmp    0x10(%rsp),%eax
    400c:	74 5a                	je     4068 <jerasure_invert_matrix+0x4b8>
      rs2 = j*cols;
      if (mat[rs2+i] != 0) {
    400e:	45 8b 2c 24          	mov    (%r12),%r13d
    4012:	45 85 ed             	test   %r13d,%r13d
    4015:	74 d9                	je     3ff0 <jerasure_invert_matrix+0x440>
        tmp = mat[rs2+i];
        mat[rs2+i] = 0; 
    4017:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    401c:	41 c7 04 24 00 00 00 	movl   $0x0,(%r12)
    4023:	00 
        for (k = 0; k < cols; k++) {
    4024:	4c 8d 34 28          	lea    (%rax,%rbp,1),%r14
    4028:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    402f:	00 
          inv[rs2+k] ^= galois_single_multiply(tmp, inv[row_start+k], w);
    4030:	43 8b 34 be          	mov    (%r14,%r15,4),%esi
    4034:	89 da                	mov    %ebx,%edx
    4036:	44 89 ef             	mov    %r13d,%edi
    4039:	e8 22 46 00 00       	callq  8660 <galois_single_multiply>
    403e:	41 31 06             	xor    %eax,(%r14)
        for (k = 0; k < cols; k++) {
    4041:	49 83 c6 04          	add    $0x4,%r14
    4045:	4c 39 f5             	cmp    %r14,%rbp
    4048:	75 e6                	jne    4030 <jerasure_invert_matrix+0x480>
    404a:	eb a4                	jmp    3ff0 <jerasure_invert_matrix+0x440>
      if (j == rows) return -1;
    404c:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
        }
      }
    }
  }
  return 0;
}
    4051:	48 81 c4 a8 00 00 00 	add    $0xa8,%rsp
    4058:	5b                   	pop    %rbx
    4059:	5d                   	pop    %rbp
    405a:	41 5c                	pop    %r12
    405c:	41 5d                	pop    %r13
    405e:	41 5e                	pop    %r14
    4060:	41 5f                	pop    %r15
    4062:	c3                   	retq   
    4063:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    4068:	44 8d 78 ff          	lea    -0x1(%rax),%r15d
    for (j = 0; j < i; j++) {
    406c:	44 89 7c 24 10       	mov    %r15d,0x10(%rsp)
    4071:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    4076:	48 83 6c 24 38 04    	subq   $0x4,0x38(%rsp)
    407c:	8b 44 24 10          	mov    0x10(%rsp),%eax
    4080:	48 01 4c 24 28       	add    %rcx,0x28(%rsp)
    4085:	85 c0                	test   %eax,%eax
    4087:	0f 8f 45 ff ff ff    	jg     3fd2 <jerasure_invert_matrix+0x422>
  for (i = rows-1; i >= 0; i--) {
    408d:	44 8b 7c 24 10       	mov    0x10(%rsp),%r15d
    4092:	41 ff cf             	dec    %r15d
    4095:	79 d5                	jns    406c <jerasure_invert_matrix+0x4bc>
  return 0;
    4097:	31 c0                	xor    %eax,%eax
    4099:	eb b6                	jmp    4051 <jerasure_invert_matrix+0x4a1>
      for (j = i+1; j < rows && mat[cols*j+i] == 0; j++) ;
    409b:	8b 44 24 18          	mov    0x18(%rsp),%eax
    409f:	e9 f2 fd ff ff       	jmpq   3e96 <jerasure_invert_matrix+0x2e6>
    40a4:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    40ab:	00 00 00 00 
    40af:	90                   	nop

00000000000040b0 <jerasure_make_decoding_matrix>:
{
    40b0:	f3 0f 1e fa          	endbr64 
    40b4:	41 57                	push   %r15
    40b6:	41 56                	push   %r14
    40b8:	4d 89 ce             	mov    %r9,%r14
    40bb:	41 55                	push   %r13
    40bd:	41 89 d5             	mov    %edx,%r13d
    40c0:	41 54                	push   %r12
    40c2:	41 89 fc             	mov    %edi,%r12d
    40c5:	0f af ff             	imul   %edi,%edi
    40c8:	55                   	push   %rbp
    40c9:	48 63 ff             	movslq %edi,%rdi
    40cc:	53                   	push   %rbx
    40cd:	48 c1 e7 02          	shl    $0x2,%rdi
    40d1:	48 83 ec 08          	sub    $0x8,%rsp
    40d5:	4c 8b 7c 24 40       	mov    0x40(%rsp),%r15
  for (i = 0; j < k; i++) {
    40da:	45 85 e4             	test   %r12d,%r12d
    40dd:	0f 8e f1 00 00 00    	jle    41d4 <jerasure_make_decoding_matrix+0x124>
    40e3:	48 89 cb             	mov    %rcx,%rbx
    40e6:	31 c0                	xor    %eax,%eax
  j = 0;
    40e8:	31 d2                	xor    %edx,%edx
    40ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    if (erased[i] == 0) {
    40f0:	41 8b 0c 80          	mov    (%r8,%rax,4),%ecx
    40f4:	85 c9                	test   %ecx,%ecx
    40f6:	75 09                	jne    4101 <jerasure_make_decoding_matrix+0x51>
      dm_ids[j] = i;
    40f8:	48 63 ca             	movslq %edx,%rcx
    40fb:	41 89 04 8f          	mov    %eax,(%r15,%rcx,4)
      j++;
    40ff:	ff c2                	inc    %edx
  for (i = 0; j < k; i++) {
    4101:	48 ff c0             	inc    %rax
    4104:	44 39 e2             	cmp    %r12d,%edx
    4107:	7c e7                	jl     40f0 <jerasure_make_decoding_matrix+0x40>
  tmpmat = talloc(int, k*k);
    4109:	e8 f2 d2 ff ff       	callq  1400 <malloc@plt>
    410e:	48 89 c5             	mov    %rax,%rbp
  if (tmpmat == NULL) { return -1; }
    4111:	48 85 c0             	test   %rax,%rax
    4114:	0f 84 c7 00 00 00    	je     41e1 <jerasure_make_decoding_matrix+0x131>
    411a:	41 8d 54 24 ff       	lea    -0x1(%r12),%edx
    411f:	48 8d 04 95 00 00 00 	lea    0x0(,%rdx,4),%rax
    4126:	00 
    4127:	4d 63 cc             	movslq %r12d,%r9
    412a:	4d 89 fa             	mov    %r15,%r10
    412d:	49 c1 e1 02          	shl    $0x2,%r9
    4131:	48 89 ef             	mov    %rbp,%rdi
    4134:	4c 8d 44 05 04       	lea    0x4(%rbp,%rax,1),%r8
    4139:	4e 8d 5c 38 04       	lea    0x4(%rax,%r15,1),%r11
    413e:	31 c9                	xor    %ecx,%ecx
    if (dm_ids[i] < k) {
    4140:	41 8b 32             	mov    (%r10),%esi
    4143:	44 39 e6             	cmp    %r12d,%esi
    4146:	7d 68                	jge    41b0 <jerasure_make_decoding_matrix+0x100>
    4148:	48 89 f8             	mov    %rdi,%rax
    414b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
      for (j = 0; j < k; j++) tmpmat[i*k+j] = 0;
    4150:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
    4156:	48 83 c0 04          	add    $0x4,%rax
    415a:	4c 39 c0             	cmp    %r8,%rax
    415d:	75 f1                	jne    4150 <jerasure_make_decoding_matrix+0xa0>
      tmpmat[i*k+dm_ids[i]] = 1;
    415f:	01 ce                	add    %ecx,%esi
    4161:	48 63 f6             	movslq %esi,%rsi
    4164:	c7 44 b5 00 01 00 00 	movl   $0x1,0x0(%rbp,%rsi,4)
    416b:	00 
  for (i = 0; i < k; i++) {
    416c:	49 83 c2 04          	add    $0x4,%r10
    4170:	44 01 e1             	add    %r12d,%ecx
    4173:	4c 01 cf             	add    %r9,%rdi
    4176:	4d 01 c8             	add    %r9,%r8
    4179:	4d 39 da             	cmp    %r11,%r10
    417c:	75 c2                	jne    4140 <jerasure_make_decoding_matrix+0x90>
  i = jerasure_invert_matrix(tmpmat, decoding_matrix, k, w);
    417e:	44 89 e2             	mov    %r12d,%edx
    4181:	48 89 ef             	mov    %rbp,%rdi
    4184:	44 89 e9             	mov    %r13d,%ecx
    4187:	4c 89 f6             	mov    %r14,%rsi
    418a:	e8 21 fa ff ff       	callq  3bb0 <jerasure_invert_matrix>
  free(tmpmat);
    418f:	48 89 ef             	mov    %rbp,%rdi
  i = jerasure_invert_matrix(tmpmat, decoding_matrix, k, w);
    4192:	41 89 c4             	mov    %eax,%r12d
  free(tmpmat);
    4195:	e8 e6 d0 ff ff       	callq  1280 <free@plt>
}
    419a:	48 83 c4 08          	add    $0x8,%rsp
    419e:	5b                   	pop    %rbx
    419f:	5d                   	pop    %rbp
    41a0:	44 89 e0             	mov    %r12d,%eax
    41a3:	41 5c                	pop    %r12
    41a5:	41 5d                	pop    %r13
    41a7:	41 5e                	pop    %r14
    41a9:	41 5f                	pop    %r15
    41ab:	c3                   	retq   
    41ac:	0f 1f 40 00          	nopl   0x0(%rax)
        tmpmat[i*k+j] = matrix[(dm_ids[i]-k)*k+j];
    41b0:	44 29 e6             	sub    %r12d,%esi
    41b3:	41 0f af f4          	imul   %r12d,%esi
    41b7:	31 c0                	xor    %eax,%eax
    41b9:	48 63 f6             	movslq %esi,%rsi
    41bc:	4c 8d 3c b3          	lea    (%rbx,%rsi,4),%r15
    41c0:	41 8b 34 87          	mov    (%r15,%rax,4),%esi
    41c4:	89 34 87             	mov    %esi,(%rdi,%rax,4)
      for (j = 0; j < k; j++) {
    41c7:	48 89 c6             	mov    %rax,%rsi
    41ca:	48 ff c0             	inc    %rax
    41cd:	48 39 f2             	cmp    %rsi,%rdx
    41d0:	75 ee                	jne    41c0 <jerasure_make_decoding_matrix+0x110>
    41d2:	eb 98                	jmp    416c <jerasure_make_decoding_matrix+0xbc>
  tmpmat = talloc(int, k*k);
    41d4:	e8 27 d2 ff ff       	callq  1400 <malloc@plt>
    41d9:	48 89 c5             	mov    %rax,%rbp
  if (tmpmat == NULL) { return -1; }
    41dc:	48 85 c0             	test   %rax,%rax
    41df:	75 9d                	jne    417e <jerasure_make_decoding_matrix+0xce>
    41e1:	41 83 cc ff          	or     $0xffffffff,%r12d
    41e5:	eb b3                	jmp    419a <jerasure_make_decoding_matrix+0xea>
    41e7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    41ee:	00 00 

00000000000041f0 <jerasure_invertible_matrix>:

int jerasure_invertible_matrix(int *mat, int rows, int w)
{
    41f0:	f3 0f 1e fa          	endbr64 
    41f4:	41 57                	push   %r15
    41f6:	41 56                	push   %r14
    41f8:	41 55                	push   %r13
    41fa:	41 54                	push   %r12
    41fc:	55                   	push   %rbp
    41fd:	53                   	push   %rbx
    41fe:	48 83 ec 78          	sub    $0x78,%rsp
    4202:	48 89 7c 24 28       	mov    %rdi,0x28(%rsp)
    4207:	89 74 24 48          	mov    %esi,0x48(%rsp)
  int row_start, tmp, inverse;
 
  cols = rows;

  /* First -- convert into upper triangular  */
  for (i = 0; i < cols; i++) {
    420b:	85 f6                	test   %esi,%esi
    420d:	0f 8e 88 02 00 00    	jle    449b <jerasure_invertible_matrix+0x2ab>
    4213:	48 63 44 24 48       	movslq 0x48(%rsp),%rax
    4218:	48 c7 44 24 20 00 00 	movq   $0x0,0x20(%rsp)
    421f:	00 00 
    4221:	48 8d 1c 85 04 00 00 	lea    0x4(,%rax,4),%rbx
    4228:	00 
    4229:	48 89 5c 24 68       	mov    %rbx,0x68(%rsp)
    422e:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
    4233:	48 89 c1             	mov    %rax,%rcx
    4236:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    423b:	48 c1 e0 02          	shl    $0x2,%rax
    423f:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    4244:	8d 41 ff             	lea    -0x1(%rcx),%eax
    4247:	48 8d 4b 04          	lea    0x4(%rbx),%rcx
    424b:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    4250:	48 8d 04 81          	lea    (%rcx,%rax,4),%rax
    4254:	48 89 5c 24 40       	mov    %rbx,0x40(%rsp)
    4259:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
    425e:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    4263:	48 89 5c 24 50       	mov    %rbx,0x50(%rsp)
    4268:	c7 44 24 4c 00 00 00 	movl   $0x0,0x4c(%rsp)
    426f:	00 
    4270:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
    4277:	00 00 
    4279:	41 89 d5             	mov    %edx,%r13d
    427c:	0f 1f 40 00          	nopl   0x0(%rax)
    row_start = cols*i;

    /* Swap rows if we ave a zero i,i element.  If we can't swap, then the 
       matrix was not invertible  */

    if (mat[row_start+i] == 0) { 
    4280:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    4285:	8b 30                	mov    (%rax),%esi
    4287:	8b 44 24 30          	mov    0x30(%rsp),%eax
    428b:	ff c0                	inc    %eax
    428d:	89 44 24 08          	mov    %eax,0x8(%rsp)
    4291:	85 f6                	test   %esi,%esi
    4293:	0f 84 2f 01 00 00    	je     43c8 <jerasure_invertible_matrix+0x1d8>
      }
    }
 
    /* Multiply the row by 1/element i,i  */
    tmp = mat[row_start+i];
    if (tmp != 1) {
    4299:	83 fe 01             	cmp    $0x1,%esi
    429c:	0f 85 b1 01 00 00    	jne    4453 <jerasure_invertible_matrix+0x263>
      }
    }

    /* Now for each j>i, add A_ji*Ai to Aj  */
    k = row_start+i;
    for (j = i+1; j != cols; j++) {
    42a2:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
    42a7:	48 39 5c 24 30       	cmp    %rbx,0x30(%rsp)
    42ac:	0f 84 e9 01 00 00    	je     449b <jerasure_invertible_matrix+0x2ab>
    42b2:	8b 5c 24 48          	mov    0x48(%rsp),%ebx
    42b6:	48 8b 6c 24 40       	mov    0x40(%rsp),%rbp
    42bb:	01 5c 24 4c          	add    %ebx,0x4c(%rsp)
    42bf:	48 03 6c 24 10       	add    0x10(%rsp),%rbp
    42c4:	8b 44 24 4c          	mov    0x4c(%rsp),%eax
    42c8:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    42cc:	0f 1f 40 00          	nopl   0x0(%rax)
      k += cols;
      if (mat[k] != 0) {
    42d0:	44 8b 65 00          	mov    0x0(%rbp),%r12d
    42d4:	45 85 e4             	test   %r12d,%r12d
    42d7:	74 52                	je     432b <jerasure_invertible_matrix+0x13b>
        if (mat[k] == 1) {
    42d9:	41 83 fc 01          	cmp    $0x1,%r12d
    42dd:	0f 84 a5 00 00 00    	je     4388 <jerasure_invertible_matrix+0x198>
    42e3:	48 63 44 24 0c       	movslq 0xc(%rsp),%rax
    42e8:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
    42ed:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
          }
        } else {
          tmp = mat[k];
          rs2 = cols*j;
          for (x = 0; x < cols; x++) {
            mat[rs2+x] ^= galois_single_multiply(tmp, mat[row_start+x], w);
    42f2:	4c 8b 7c 24 20       	mov    0x20(%rsp),%r15
    42f7:	4c 8d 34 83          	lea    (%rbx,%rax,4),%r14
    42fb:	48 8b 5c 24 38       	mov    0x38(%rsp),%rbx
    4300:	48 8d 14 08          	lea    (%rax,%rcx,1),%rdx
    4304:	48 8d 1c 93          	lea    (%rbx,%rdx,4),%rbx
    4308:	49 29 c7             	sub    %rax,%r15
    430b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    4310:	43 8b 34 be          	mov    (%r14,%r15,4),%esi
    4314:	44 89 ea             	mov    %r13d,%edx
    4317:	44 89 e7             	mov    %r12d,%edi
    431a:	e8 41 43 00 00       	callq  8660 <galois_single_multiply>
    431f:	41 31 06             	xor    %eax,(%r14)
          for (x = 0; x < cols; x++) {
    4322:	49 83 c6 04          	add    $0x4,%r14
    4326:	4c 39 f3             	cmp    %r14,%rbx
    4329:	75 e5                	jne    4310 <jerasure_invertible_matrix+0x120>
    for (j = i+1; j != cols; j++) {
    432b:	ff 44 24 08          	incl   0x8(%rsp)
    432f:	8b 7c 24 48          	mov    0x48(%rsp),%edi
    4333:	48 03 6c 24 10       	add    0x10(%rsp),%rbp
    4338:	01 7c 24 0c          	add    %edi,0xc(%rsp)
    433c:	8b 44 24 08          	mov    0x8(%rsp),%eax
    4340:	39 c7                	cmp    %eax,%edi
    4342:	75 8c                	jne    42d0 <jerasure_invertible_matrix+0xe0>
  for (i = 0; i < cols; i++) {
    4344:	48 8b 54 24 68       	mov    0x68(%rsp),%rdx
    4349:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    434e:	48 01 54 24 40       	add    %rdx,0x40(%rsp)
    4353:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
    4358:	48 8b 54 24 60       	mov    0x60(%rsp),%rdx
    435d:	48 01 4c 24 50       	add    %rcx,0x50(%rsp)
    4362:	48 01 54 24 20       	add    %rdx,0x20(%rsp)
    4367:	48 01 4c 24 58       	add    %rcx,0x58(%rsp)
    436c:	48 8d 43 01          	lea    0x1(%rbx),%rax
    4370:	48 3b 5c 24 18       	cmp    0x18(%rsp),%rbx
    4375:	0f 84 20 01 00 00    	je     449b <jerasure_invertible_matrix+0x2ab>
    437b:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    4380:	e9 fb fe ff ff       	jmpq   4280 <jerasure_invertible_matrix+0x90>
    4385:	0f 1f 00             	nopl   (%rax)
    4388:	48 63 54 24 0c       	movslq 0xc(%rsp),%rdx
    438d:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
    4392:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    4397:	48 8d 0c 1a          	lea    (%rdx,%rbx,1),%rcx
    439b:	48 8b 5c 24 38       	mov    0x38(%rsp),%rbx
    43a0:	48 8d 04 90          	lea    (%rax,%rdx,4),%rax
    43a4:	48 8d 34 8b          	lea    (%rbx,%rcx,4),%rsi
            mat[rs2+x] ^= mat[row_start+x];
    43a8:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    43ad:	48 29 d1             	sub    %rdx,%rcx
    43b0:	8b 14 88             	mov    (%rax,%rcx,4),%edx
    43b3:	31 10                	xor    %edx,(%rax)
          for (x = 0; x < cols; x++) {
    43b5:	48 83 c0 04          	add    $0x4,%rax
    43b9:	48 39 f0             	cmp    %rsi,%rax
    43bc:	75 f2                	jne    43b0 <jerasure_invertible_matrix+0x1c0>
    43be:	e9 68 ff ff ff       	jmpq   432b <jerasure_invertible_matrix+0x13b>
    43c3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
      for (j = i+1; j < rows && mat[cols*j+i] == 0; j++) ;
    43c8:	44 8b 44 24 48       	mov    0x48(%rsp),%r8d
    43cd:	89 c7                	mov    %eax,%edi
    43cf:	41 39 c0             	cmp    %eax,%r8d
    43d2:	0f 8e d9 00 00 00    	jle    44b1 <jerasure_invertible_matrix+0x2c1>
    43d8:	8b 44 24 4c          	mov    0x4c(%rsp),%eax
    43dc:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
    43e1:	44 01 c0             	add    %r8d,%eax
    43e4:	48 63 d0             	movslq %eax,%rdx
    43e7:	48 03 54 24 30       	add    0x30(%rsp),%rdx
    43ec:	4c 8b 4c 24 10       	mov    0x10(%rsp),%r9
    43f1:	48 8d 0c 93          	lea    (%rbx,%rdx,4),%rcx
    43f5:	eb 1a                	jmp    4411 <jerasure_invertible_matrix+0x221>
    43f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    43fe:	00 00 
    4400:	ff c7                	inc    %edi
    4402:	44 01 c0             	add    %r8d,%eax
    4405:	4c 01 c9             	add    %r9,%rcx
    4408:	41 39 f8             	cmp    %edi,%r8d
    440b:	0f 84 8f 00 00 00    	je     44a0 <jerasure_invertible_matrix+0x2b0>
    4411:	8b 31                	mov    (%rcx),%esi
    4413:	89 c2                	mov    %eax,%edx
    4415:	85 f6                	test   %esi,%esi
    4417:	74 e7                	je     4400 <jerasure_invertible_matrix+0x210>
      for (k = 0; k < cols; k++) {
    4419:	48 63 d2             	movslq %edx,%rdx
      for (j = i+1; j < rows && mat[cols*j+i] == 0; j++) ;
    441c:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    4421:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    4426:	48 2b 54 24 20       	sub    0x20(%rsp),%rdx
    442b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
        tmp = mat[row_start+k];
    4430:	8b 08                	mov    (%rax),%ecx
        mat[row_start+k] = mat[rs2+k];
    4432:	8b 34 90             	mov    (%rax,%rdx,4),%esi
    4435:	89 30                	mov    %esi,(%rax)
        mat[rs2+k] = tmp;
    4437:	89 0c 90             	mov    %ecx,(%rax,%rdx,4)
      for (k = 0; k < cols; k++) {
    443a:	48 83 c0 04          	add    $0x4,%rax
    443e:	48 39 f8             	cmp    %rdi,%rax
    4441:	75 ed                	jne    4430 <jerasure_invertible_matrix+0x240>
    4443:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    4448:	8b 30                	mov    (%rax),%esi
    if (tmp != 1) {
    444a:	83 fe 01             	cmp    $0x1,%esi
    444d:	0f 84 4f fe ff ff    	je     42a2 <jerasure_invertible_matrix+0xb2>
      inverse = galois_single_divide(1, tmp, w);
    4453:	44 89 ea             	mov    %r13d,%edx
    4456:	bf 01 00 00 00       	mov    $0x1,%edi
    445b:	e8 20 42 00 00       	callq  8680 <galois_single_divide>
    4460:	4c 8b 64 24 50       	mov    0x50(%rsp),%r12
    4465:	48 8b 5c 24 58       	mov    0x58(%rsp),%rbx
    446a:	89 c5                	mov    %eax,%ebp
      for (j = 0; j < cols; j++) { 
    446c:	0f 1f 40 00          	nopl   0x0(%rax)
        mat[row_start+j] = galois_single_multiply(mat[row_start+j], inverse, w);
    4470:	41 8b 3c 24          	mov    (%r12),%edi
    4474:	44 89 ea             	mov    %r13d,%edx
    4477:	89 ee                	mov    %ebp,%esi
    4479:	e8 e2 41 00 00       	callq  8660 <galois_single_multiply>
    447e:	41 89 04 24          	mov    %eax,(%r12)
      for (j = 0; j < cols; j++) { 
    4482:	49 83 c4 04          	add    $0x4,%r12
    4486:	49 39 dc             	cmp    %rbx,%r12
    4489:	75 e5                	jne    4470 <jerasure_invertible_matrix+0x280>
    for (j = i+1; j != cols; j++) {
    448b:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
    4490:	48 39 5c 24 30       	cmp    %rbx,0x30(%rsp)
    4495:	0f 85 17 fe ff ff    	jne    42b2 <jerasure_invertible_matrix+0xc2>
          }
        }
      }
    }
  }
  return 1;
    449b:	be 01 00 00 00       	mov    $0x1,%esi
}
    44a0:	48 83 c4 78          	add    $0x78,%rsp
    44a4:	5b                   	pop    %rbx
    44a5:	5d                   	pop    %rbp
    44a6:	41 5c                	pop    %r12
    44a8:	41 5d                	pop    %r13
    44aa:	41 5e                	pop    %r14
    44ac:	89 f0                	mov    %esi,%eax
    44ae:	41 5f                	pop    %r15
    44b0:	c3                   	retq   
      if (j == rows) return 0;
    44b1:	74 ed                	je     44a0 <jerasure_invertible_matrix+0x2b0>
    44b3:	8b 54 24 48          	mov    0x48(%rsp),%edx
    44b7:	0f af 54 24 08       	imul   0x8(%rsp),%edx
    44bc:	e9 58 ff ff ff       	jmpq   4419 <jerasure_invertible_matrix+0x229>
    44c1:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    44c8:	00 00 00 00 
    44cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000044d0 <jerasure_erasures_to_erased>:

/* Converts a list-style version of the erasures into an array of k+m elements
   where the element = 1 if the index has been erased, and zero otherwise */

int *jerasure_erasures_to_erased(int k, int m, int *erasures)
{
    44d0:	f3 0f 1e fa          	endbr64 
    44d4:	41 54                	push   %r12
    44d6:	41 89 fc             	mov    %edi,%r12d
    44d9:	55                   	push   %rbp
  int td;
  int t_non_erased;
  int *erased;
  int i;

  td = k+m;
    44da:	8d 2c 37             	lea    (%rdi,%rsi,1),%ebp
  erased = talloc(int, td);
    44dd:	48 63 fd             	movslq %ebp,%rdi
{
    44e0:	53                   	push   %rbx
  erased = talloc(int, td);
    44e1:	48 c1 e7 02          	shl    $0x2,%rdi
{
    44e5:	48 89 d3             	mov    %rdx,%rbx
  erased = talloc(int, td);
    44e8:	e8 13 cf ff ff       	callq  1400 <malloc@plt>
  if (erased == NULL) return NULL;
    44ed:	48 85 c0             	test   %rax,%rax
    44f0:	74 59                	je     454b <jerasure_erasures_to_erased+0x7b>
  t_non_erased = td;

  for (i = 0; i < td; i++) erased[i] = 0;
    44f2:	85 ed                	test   %ebp,%ebp
    44f4:	7e 21                	jle    4517 <jerasure_erasures_to_erased+0x47>
    44f6:	8d 4d ff             	lea    -0x1(%rbp),%ecx
    44f9:	48 89 c2             	mov    %rax,%rdx
    44fc:	48 8d 4c 88 04       	lea    0x4(%rax,%rcx,4),%rcx
    4501:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    4508:	c7 02 00 00 00 00    	movl   $0x0,(%rdx)
    450e:	48 83 c2 04          	add    $0x4,%rdx
    4512:	48 39 ca             	cmp    %rcx,%rdx
    4515:	75 f1                	jne    4508 <jerasure_erasures_to_erased+0x38>

  for (i = 0; erasures[i] != -1; i++) {
    4517:	48 63 0b             	movslq (%rbx),%rcx
    451a:	83 f9 ff             	cmp    $0xffffffff,%ecx
    451d:	74 2c                	je     454b <jerasure_erasures_to_erased+0x7b>
    451f:	48 8d 53 04          	lea    0x4(%rbx),%rdx
    4523:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    if (erased[erasures[i]] == 0) {
    4528:	48 8d 0c 88          	lea    (%rax,%rcx,4),%rcx
    452c:	8b 31                	mov    (%rcx),%esi
    452e:	85 f6                	test   %esi,%esi
    4530:	75 0d                	jne    453f <jerasure_erasures_to_erased+0x6f>
      erased[erasures[i]] = 1;
      t_non_erased--;
    4532:	ff cd                	dec    %ebp
      erased[erasures[i]] = 1;
    4534:	c7 01 01 00 00 00    	movl   $0x1,(%rcx)
      if (t_non_erased < k) {
    453a:	41 39 ec             	cmp    %ebp,%r12d
    453d:	7f 11                	jg     4550 <jerasure_erasures_to_erased+0x80>
  for (i = 0; erasures[i] != -1; i++) {
    453f:	48 63 0a             	movslq (%rdx),%rcx
    4542:	48 83 c2 04          	add    $0x4,%rdx
    4546:	83 f9 ff             	cmp    $0xffffffff,%ecx
    4549:	75 dd                	jne    4528 <jerasure_erasures_to_erased+0x58>
        return NULL;
      }
    }
  }
  return erased;
}
    454b:	5b                   	pop    %rbx
    454c:	5d                   	pop    %rbp
    454d:	41 5c                	pop    %r12
    454f:	c3                   	retq   
        free(erased);
    4550:	48 89 c7             	mov    %rax,%rdi
    4553:	e8 28 cd ff ff       	callq  1280 <free@plt>
}
    4558:	5b                   	pop    %rbx
    4559:	5d                   	pop    %rbp
        return NULL;
    455a:	31 c0                	xor    %eax,%eax
}
    455c:	41 5c                	pop    %r12
    455e:	c3                   	retq   
    455f:	90                   	nop

0000000000004560 <set_up_ptrs_for_scheduled_decoding>:

  return 0;
}

static char **set_up_ptrs_for_scheduled_decoding(int k, int m, int *erasures, char **data_ptrs, char **coding_ptrs)
{
    4560:	41 57                	push   %r15
    4562:	41 56                	push   %r14
    4564:	41 89 f6             	mov    %esi,%r14d
    4567:	41 55                	push   %r13
    4569:	4d 89 c5             	mov    %r8,%r13
    456c:	41 54                	push   %r12
    456e:	49 89 cc             	mov    %rcx,%r12
    4571:	55                   	push   %rbp
    4572:	53                   	push   %rbx
    4573:	48 63 df             	movslq %edi,%rbx
  cdf = 0;
  for (i = 0; erasures[i] != -1; i++) {
    if (erasures[i] < k) ddf++; else cdf++;
  }
  
  erased = jerasure_erasures_to_erased(k, m, erasures);
    4576:	89 df                	mov    %ebx,%edi
{
    4578:	48 83 ec 18          	sub    $0x18,%rsp
  erased = jerasure_erasures_to_erased(k, m, erasures);
    457c:	e8 4f ff ff ff       	callq  44d0 <jerasure_erasures_to_erased>
  if (erased == NULL) return NULL;
    4581:	48 85 c0             	test   %rax,%rax
    4584:	0f 84 f5 00 00 00    	je     467f <set_up_ptrs_for_scheduled_decoding+0x11f>
       The array ind_to_row_ids contains the row_id of drive i.
  
       However, we're going to set row_ids and ind_to_row in a different procedure.
   */
         
  ptrs = talloc(char *, k+m);
    458a:	42 8d 0c 33          	lea    (%rbx,%r14,1),%ecx
    458e:	48 63 f9             	movslq %ecx,%rdi
    4591:	48 c1 e7 03          	shl    $0x3,%rdi
    4595:	89 4c 24 0c          	mov    %ecx,0xc(%rsp)
    4599:	48 89 c5             	mov    %rax,%rbp
    459c:	e8 5f ce ff ff       	callq  1400 <malloc@plt>

  j = k;
  x = k;
  for (i = 0; i < k; i++) {
    45a1:	85 db                	test   %ebx,%ebx
    45a3:	8b 4c 24 0c          	mov    0xc(%rsp),%ecx
  ptrs = talloc(char *, k+m);
    45a7:	49 89 c7             	mov    %rax,%r15
  for (i = 0; i < k; i++) {
    45aa:	0f 8e d4 00 00 00    	jle    4684 <set_up_ptrs_for_scheduled_decoding+0x124>
    45b0:	8d 73 ff             	lea    -0x1(%rbx),%esi
    45b3:	41 89 da             	mov    %ebx,%r10d
    45b6:	41 89 d9             	mov    %ebx,%r9d
    45b9:	31 ff                	xor    %edi,%edi
    45bb:	eb 13                	jmp    45d0 <set_up_ptrs_for_scheduled_decoding+0x70>
    45bd:	0f 1f 00             	nopl   (%rax)
    if (erased[i] == 0) {
      ptrs[i] = data_ptrs[i];
    45c0:	49 89 04 ff          	mov    %rax,(%r15,%rdi,8)
  for (i = 0; i < k; i++) {
    45c4:	48 8d 47 01          	lea    0x1(%rdi),%rax
    45c8:	48 39 fe             	cmp    %rdi,%rsi
    45cb:	74 5a                	je     4627 <set_up_ptrs_for_scheduled_decoding+0xc7>
    45cd:	48 89 c7             	mov    %rax,%rdi
    if (erased[i] == 0) {
    45d0:	8b 54 bd 00          	mov    0x0(%rbp,%rdi,4),%edx
    45d4:	49 8b 04 fc          	mov    (%r12,%rdi,8),%rax
    45d8:	85 d2                	test   %edx,%edx
    45da:	74 e4                	je     45c0 <set_up_ptrs_for_scheduled_decoding+0x60>
    } else {
      while (erased[j]) j++;
    45dc:	4d 63 c1             	movslq %r9d,%r8
    45df:	46 8b 5c 85 00       	mov    0x0(%rbp,%r8,4),%r11d
    45e4:	41 8d 51 01          	lea    0x1(%r9),%edx
    45e8:	48 63 d2             	movslq %edx,%rdx
    45eb:	45 85 db             	test   %r11d,%r11d
    45ee:	74 10                	je     4600 <set_up_ptrs_for_scheduled_decoding+0xa0>
    45f0:	41 89 d1             	mov    %edx,%r9d
    45f3:	48 ff c2             	inc    %rdx
    45f6:	44 8b 44 95 fc       	mov    -0x4(%rbp,%rdx,4),%r8d
    45fb:	45 85 c0             	test   %r8d,%r8d
    45fe:	75 f0                	jne    45f0 <set_up_ptrs_for_scheduled_decoding+0x90>
      ptrs[i] = coding_ptrs[j-k];
    4600:	44 89 ca             	mov    %r9d,%edx
    4603:	29 da                	sub    %ebx,%edx
    4605:	48 63 d2             	movslq %edx,%rdx
    4608:	49 8b 54 d5 00       	mov    0x0(%r13,%rdx,8),%rdx
      j++;
    460d:	41 ff c1             	inc    %r9d
      ptrs[i] = coding_ptrs[j-k];
    4610:	49 89 14 ff          	mov    %rdx,(%r15,%rdi,8)
      ptrs[x] = data_ptrs[i];
    4614:	49 63 d2             	movslq %r10d,%rdx
    4617:	49 89 04 d7          	mov    %rax,(%r15,%rdx,8)
      x++;
    461b:	41 ff c2             	inc    %r10d
  for (i = 0; i < k; i++) {
    461e:	48 8d 47 01          	lea    0x1(%rdi),%rax
    4622:	48 39 fe             	cmp    %rdi,%rsi
    4625:	75 a6                	jne    45cd <set_up_ptrs_for_scheduled_decoding+0x6d>
    }
  }
  for (i = k; i < k+m; i++) {
    4627:	39 d9                	cmp    %ebx,%ecx
    4629:	7e 3a                	jle    4665 <set_up_ptrs_for_scheduled_decoding+0x105>
    462b:	41 8d 76 ff          	lea    -0x1(%r14),%esi
    462f:	48 8d 4c 9d 00       	lea    0x0(%rbp,%rbx,4),%rcx
    4634:	31 d2                	xor    %edx,%edx
    4636:	eb 0b                	jmp    4643 <set_up_ptrs_for_scheduled_decoding+0xe3>
    4638:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    463f:	00 
    4640:	48 89 c2             	mov    %rax,%rdx
    if (erased[i]) {
    4643:	8b 04 91             	mov    (%rcx,%rdx,4),%eax
    4646:	85 c0                	test   %eax,%eax
    4648:	74 12                	je     465c <set_up_ptrs_for_scheduled_decoding+0xfc>
      ptrs[x] = coding_ptrs[i-k];
    464a:	48 63 fa             	movslq %edx,%rdi
    464d:	49 8b 7c fd 00       	mov    0x0(%r13,%rdi,8),%rdi
    4652:	49 63 c2             	movslq %r10d,%rax
    4655:	49 89 3c c7          	mov    %rdi,(%r15,%rax,8)
      x++;
    4659:	41 ff c2             	inc    %r10d
  for (i = k; i < k+m; i++) {
    465c:	48 8d 42 01          	lea    0x1(%rdx),%rax
    4660:	48 39 d6             	cmp    %rdx,%rsi
    4663:	75 db                	jne    4640 <set_up_ptrs_for_scheduled_decoding+0xe0>
    }
  }
  free(erased);
    4665:	48 89 ef             	mov    %rbp,%rdi
    4668:	e8 13 cc ff ff       	callq  1280 <free@plt>
  return ptrs;
}
    466d:	48 83 c4 18          	add    $0x18,%rsp
    4671:	5b                   	pop    %rbx
    4672:	5d                   	pop    %rbp
    4673:	41 5c                	pop    %r12
    4675:	41 5d                	pop    %r13
    4677:	41 5e                	pop    %r14
    4679:	4c 89 f8             	mov    %r15,%rax
    467c:	41 5f                	pop    %r15
    467e:	c3                   	retq   
  if (erased == NULL) return NULL;
    467f:	45 31 ff             	xor    %r15d,%r15d
    4682:	eb e9                	jmp    466d <set_up_ptrs_for_scheduled_decoding+0x10d>
  for (i = 0; i < k; i++) {
    4684:	41 89 da             	mov    %ebx,%r10d
    4687:	eb 9e                	jmp    4627 <set_up_ptrs_for_scheduled_decoding+0xc7>
    4689:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000004690 <jerasure_free_schedule>:
{
    4690:	f3 0f 1e fa          	endbr64 
    4694:	55                   	push   %rbp
    4695:	48 89 fd             	mov    %rdi,%rbp
    4698:	53                   	push   %rbx
    4699:	48 83 ec 08          	sub    $0x8,%rsp
  for (i = 0; schedule[i][0] >= 0; i++) free(schedule[i]);
    469d:	48 8b 3f             	mov    (%rdi),%rdi
    46a0:	8b 17                	mov    (%rdi),%edx
    46a2:	85 d2                	test   %edx,%edx
    46a4:	78 1c                	js     46c2 <jerasure_free_schedule+0x32>
    46a6:	48 8d 5d 08          	lea    0x8(%rbp),%rbx
    46aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    46b0:	e8 cb cb ff ff       	callq  1280 <free@plt>
    46b5:	48 8b 3b             	mov    (%rbx),%rdi
    46b8:	48 83 c3 08          	add    $0x8,%rbx
    46bc:	8b 07                	mov    (%rdi),%eax
    46be:	85 c0                	test   %eax,%eax
    46c0:	79 ee                	jns    46b0 <jerasure_free_schedule+0x20>
  free(schedule[i]);
    46c2:	e8 b9 cb ff ff       	callq  1280 <free@plt>
}
    46c7:	48 83 c4 08          	add    $0x8,%rsp
    46cb:	5b                   	pop    %rbx
  free(schedule);
    46cc:	48 89 ef             	mov    %rbp,%rdi
}
    46cf:	5d                   	pop    %rbp
  free(schedule);
    46d0:	e9 ab cb ff ff       	jmpq   1280 <free@plt>
    46d5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    46dc:	00 00 00 00 

00000000000046e0 <jerasure_free_schedule_cache>:
{
    46e0:	f3 0f 1e fa          	endbr64 
    46e4:	41 57                	push   %r15
    46e6:	41 56                	push   %r14
    46e8:	41 55                	push   %r13
    46ea:	41 54                	push   %r12
    46ec:	55                   	push   %rbp
    46ed:	53                   	push   %rbx
    46ee:	48 83 ec 28          	sub    $0x28,%rsp
    46f2:	48 89 54 24 18       	mov    %rdx,0x18(%rsp)
  if (m != 2) {
    46f7:	83 fe 02             	cmp    $0x2,%esi
    46fa:	0f 85 88 00 00 00    	jne    4788 <jerasure_free_schedule_cache+0xa8>
  for (e1 = 0; e1 < k+m; e1++) {
    4700:	8d 47 02             	lea    0x2(%rdi),%eax
    4703:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    4707:	85 c0                	test   %eax,%eax
    4709:	7e 65                	jle    4770 <jerasure_free_schedule_cache+0x90>
    470b:	4c 63 f0             	movslq %eax,%r14
    470e:	4e 8d 3c f5 00 00 00 	lea    0x0(,%r14,8),%r15
    4715:	00 
    4716:	49 8d 47 08          	lea    0x8(%r15),%rax
    471a:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    471f:	49 f7 de             	neg    %r14
    4722:	4e 8d 24 3a          	lea    (%rdx,%r15,1),%r12
    4726:	4a 8d 5c 3a 08       	lea    0x8(%rdx,%r15,1),%rbx
    472b:	49 c1 e6 03          	shl    $0x3,%r14
    472f:	31 ed                	xor    %ebp,%ebp
    4731:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    jerasure_free_schedule(cache[e1*(k+m)+e1]);
    4738:	4a 8b 7c 33 f8       	mov    -0x8(%rbx,%r14,1),%rdi
  for (e1 = 0; e1 < k+m; e1++) {
    473d:	ff c5                	inc    %ebp
    jerasure_free_schedule(cache[e1*(k+m)+e1]);
    473f:	e8 4c ff ff ff       	callq  4690 <jerasure_free_schedule>
  for (e1 = 0; e1 < k+m; e1++) {
    4744:	3b 6c 24 0c          	cmp    0xc(%rsp),%ebp
    4748:	74 26                	je     4770 <jerasure_free_schedule_cache+0x90>
    474a:	4d 89 e5             	mov    %r12,%r13
    474d:	0f 1f 00             	nopl   (%rax)
      jerasure_free_schedule(cache[e1*(k+m)+e2]);
    4750:	49 8b 7d 00          	mov    0x0(%r13),%rdi
    4754:	49 83 c5 08          	add    $0x8,%r13
    4758:	e8 33 ff ff ff       	callq  4690 <jerasure_free_schedule>
    for (e2 = 0; e2 < e1; e2++) {
    475d:	49 39 dd             	cmp    %rbx,%r13
    4760:	75 ee                	jne    4750 <jerasure_free_schedule_cache+0x70>
    4762:	4d 01 fc             	add    %r15,%r12
    4765:	48 03 5c 24 10       	add    0x10(%rsp),%rbx
    476a:	eb cc                	jmp    4738 <jerasure_free_schedule_cache+0x58>
    476c:	0f 1f 40 00          	nopl   0x0(%rax)
  free(cache);
    4770:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
}
    4775:	48 83 c4 28          	add    $0x28,%rsp
    4779:	5b                   	pop    %rbx
    477a:	5d                   	pop    %rbp
    477b:	41 5c                	pop    %r12
    477d:	41 5d                	pop    %r13
    477f:	41 5e                	pop    %r14
    4781:	41 5f                	pop    %r15
  free(cache);
    4783:	e9 f8 ca ff ff       	jmpq   1280 <free@plt>
    4788:	48 8b 0d b1 c9 00 00 	mov    0xc9b1(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    478f:	48 8d 3d a2 5f 00 00 	lea    0x5fa2(%rip),%rdi        # a738 <__PRETTY_FUNCTION__.5741+0x5f>
    4796:	ba 2f 00 00 00       	mov    $0x2f,%edx
    479b:	be 01 00 00 00       	mov    $0x1,%esi
    47a0:	e8 cb cc ff ff       	callq  1470 <fwrite@plt>
    exit(1);
    47a5:	bf 01 00 00 00       	mov    $0x1,%edi
    47aa:	e8 b1 cc ff ff       	callq  1460 <exit@plt>
    47af:	90                   	nop

00000000000047b0 <jerasure_matrix_dotprod>:
{
    47b0:	f3 0f 1e fa          	endbr64 
    47b4:	41 57                	push   %r15
    47b6:	41 56                	push   %r14
    47b8:	41 55                	push   %r13
    47ba:	41 54                	push   %r12
    47bc:	55                   	push   %rbp
    47bd:	53                   	push   %rbx
    47be:	48 83 ec 28          	sub    $0x28,%rsp
    47c2:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
    47c7:	89 7c 24 18          	mov    %edi,0x18(%rsp)
    47cb:	44 8b 4c 24 68       	mov    0x68(%rsp),%r9d
  if (w != 1 && w != 8 && w != 16 && w != 32) {
    47d0:	83 fe 20             	cmp    $0x20,%esi
    47d3:	0f 87 af 01 00 00    	ja     4988 <jerasure_matrix_dotprod+0x1d8>
    47d9:	48 b8 02 01 01 00 01 	movabs $0x100010102,%rax
    47e0:	00 00 00 
    47e3:	48 0f a3 f0          	bt     %rsi,%rax
    47e7:	89 f5                	mov    %esi,%ebp
    47e9:	0f 83 99 01 00 00    	jae    4988 <jerasure_matrix_dotprod+0x1d8>
    47ef:	48 89 d3             	mov    %rdx,%rbx
    47f2:	49 89 cd             	mov    %rcx,%r13
  dptr = (dest_id < k) ? data_ptrs[dest_id] : coding_ptrs[dest_id-k];
    47f5:	44 3b 44 24 18       	cmp    0x18(%rsp),%r8d
    47fa:	0f 8c b0 01 00 00    	jl     49b0 <jerasure_matrix_dotprod+0x200>
    4800:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    4805:	44 2b 44 24 18       	sub    0x18(%rsp),%r8d
    480a:	4d 63 c0             	movslq %r8d,%r8
    480d:	4e 8b 34 c0          	mov    (%rax,%r8,8),%r14
  for (i = 0; i < k; i++) {
    4811:	8b 44 24 18          	mov    0x18(%rsp),%eax
    4815:	85 c0                	test   %eax,%eax
    4817:	0f 8e ab 01 00 00    	jle    49c8 <jerasure_matrix_dotprod+0x218>
    481d:	44 8d 60 ff          	lea    -0x1(%rax),%r12d
        memcpy(dptr, sptr, size);
    4821:	49 63 c9             	movslq %r9d,%rcx
  init = 0;
    4824:	31 c0                	xor    %eax,%eax
    4826:	89 6c 24 1c          	mov    %ebp,0x1c(%rsp)
        memcpy(dptr, sptr, size);
    482a:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    482f:	4c 89 f5             	mov    %r14,%rbp
  for (i = 0; i < k; i++) {
    4832:	45 31 ff             	xor    %r15d,%r15d
    4835:	4d 89 ee             	mov    %r13,%r14
    4838:	49 89 dd             	mov    %rbx,%r13
    483b:	89 c3                	mov    %eax,%ebx
    483d:	eb 0d                	jmp    484c <jerasure_matrix_dotprod+0x9c>
    483f:	90                   	nop
    4840:	49 8d 57 01          	lea    0x1(%r15),%rdx
    4844:	4d 39 fc             	cmp    %r15,%r12
    4847:	74 77                	je     48c0 <jerasure_matrix_dotprod+0x110>
    4849:	49 89 d7             	mov    %rdx,%r15
    if (matrix_row[i] == 1) {
    484c:	43 83 7c bd 00 01    	cmpl   $0x1,0x0(%r13,%r15,4)
    4852:	75 ec                	jne    4840 <jerasure_matrix_dotprod+0x90>
      if (src_ids == NULL) {
    4854:	4d 85 f6             	test   %r14,%r14
    4857:	0f 84 23 02 00 00    	je     4a80 <jerasure_matrix_dotprod+0x2d0>
      } else if (src_ids[i] < k) {
    485d:	4b 63 14 be          	movslq (%r14,%r15,4),%rdx
    4861:	3b 54 24 18          	cmp    0x18(%rsp),%edx
    4865:	0f 8d b5 01 00 00    	jge    4a20 <jerasure_matrix_dotprod+0x270>
        sptr = data_ptrs[src_ids[i]];
    486b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    4870:	48 8b 3c d0          	mov    (%rax,%rdx,8),%rdi
      if (init == 0) {
    4874:	85 db                	test   %ebx,%ebx
    4876:	0f 85 64 01 00 00    	jne    49e0 <jerasure_matrix_dotprod+0x230>
    487c:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    4881:	48 89 fe             	mov    %rdi,%rsi
    4884:	48 89 ef             	mov    %rbp,%rdi
    4887:	44 89 4c 24 68       	mov    %r9d,0x68(%rsp)
    488c:	e8 2f cb ff ff       	callq  13c0 <memcpy@plt>
        jerasure_total_memcpy_bytes += size;
    4891:	44 8b 4c 24 68       	mov    0x68(%rsp),%r9d
    4896:	c5 e9 57 d2          	vxorpd %xmm2,%xmm2,%xmm2
    489a:	c4 c1 6b 2a c1       	vcvtsi2sd %r9d,%xmm2,%xmm0
        init = 1;
    489f:	bb 01 00 00 00       	mov    $0x1,%ebx
    48a4:	49 8d 57 01          	lea    0x1(%r15),%rdx
        jerasure_total_memcpy_bytes += size;
    48a8:	c5 fb 58 05 a0 c8 00 	vaddsd 0xc8a0(%rip),%xmm0,%xmm0        # 11150 <jerasure_total_memcpy_bytes>
    48af:	00 
    48b0:	c5 fb 11 05 98 c8 00 	vmovsd %xmm0,0xc898(%rip)        # 11150 <jerasure_total_memcpy_bytes>
    48b7:	00 
  for (i = 0; i < k; i++) {
    48b8:	4d 39 fc             	cmp    %r15,%r12
    48bb:	75 8c                	jne    4849 <jerasure_matrix_dotprod+0x99>
    48bd:	0f 1f 00             	nopl   (%rax)
    48c0:	89 d8                	mov    %ebx,%eax
    48c2:	4c 89 eb             	mov    %r13,%rbx
    48c5:	4d 89 f5             	mov    %r14,%r13
    48c8:	49 89 ee             	mov    %rbp,%r14
    48cb:	8b 6c 24 1c          	mov    0x1c(%rsp),%ebp
    48cf:	41 89 c0             	mov    %eax,%r8d
    48d2:	4c 89 e8             	mov    %r13,%rax
    48d5:	4c 89 74 24 10       	mov    %r14,0x10(%rsp)
    48da:	41 89 ed             	mov    %ebp,%r13d
    48dd:	45 31 ff             	xor    %r15d,%r15d
    48e0:	48 89 dd             	mov    %rbx,%rbp
    48e3:	45 89 ce             	mov    %r9d,%r14d
    48e6:	48 89 c3             	mov    %rax,%rbx
    48e9:	eb 57                	jmp    4942 <jerasure_matrix_dotprod+0x192>
    48eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
        sptr = data_ptrs[src_ids[i]];
    48f0:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    48f5:	48 8b 3c d0          	mov    (%rax,%rdx,8),%rdi
      switch (w) {
    48f9:	41 83 fd 10          	cmp    $0x10,%r13d
    48fd:	74 76                	je     4975 <jerasure_matrix_dotprod+0x1c5>
    48ff:	41 83 fd 20          	cmp    $0x20,%r13d
    4903:	0f 84 5f 01 00 00    	je     4a68 <jerasure_matrix_dotprod+0x2b8>
    4909:	41 83 fd 08          	cmp    $0x8,%r13d
    490d:	0f 84 3d 01 00 00    	je     4a50 <jerasure_matrix_dotprod+0x2a0>
      jerasure_total_gf_bytes += size;
    4913:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
    4917:	c4 c1 73 2a c6       	vcvtsi2sd %r14d,%xmm1,%xmm0
      init = 1;
    491c:	41 b8 01 00 00 00    	mov    $0x1,%r8d
      jerasure_total_gf_bytes += size;
    4922:	c5 fb 58 05 2e c8 00 	vaddsd 0xc82e(%rip),%xmm0,%xmm0        # 11158 <jerasure_total_gf_bytes>
    4929:	00 
    492a:	c5 fb 11 05 26 c8 00 	vmovsd %xmm0,0xc826(%rip)        # 11158 <jerasure_total_gf_bytes>
    4931:	00 
  for (i = 0; i < k; i++) {
    4932:	49 8d 57 01          	lea    0x1(%r15),%rdx
    4936:	4d 39 fc             	cmp    %r15,%r12
    4939:	0f 84 89 00 00 00    	je     49c8 <jerasure_matrix_dotprod+0x218>
    493f:	49 89 d7             	mov    %rdx,%r15
    if (matrix_row[i] != 0 && matrix_row[i] != 1) {
    4942:	42 8b 74 bd 00       	mov    0x0(%rbp,%r15,4),%esi
    4947:	83 fe 01             	cmp    $0x1,%esi
    494a:	76 e6                	jbe    4932 <jerasure_matrix_dotprod+0x182>
      if (src_ids == NULL) {
    494c:	48 85 db             	test   %rbx,%rbx
    494f:	0f 84 e3 00 00 00    	je     4a38 <jerasure_matrix_dotprod+0x288>
      } else if (src_ids[i] < k) {
    4955:	4a 63 14 bb          	movslq (%rbx,%r15,4),%rdx
    4959:	3b 54 24 18          	cmp    0x18(%rsp),%edx
    495d:	7c 91                	jl     48f0 <jerasure_matrix_dotprod+0x140>
        sptr = coding_ptrs[src_ids[i]-k];
    495f:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    4964:	2b 54 24 18          	sub    0x18(%rsp),%edx
    4968:	48 63 d2             	movslq %edx,%rdx
    496b:	48 8b 3c d0          	mov    (%rax,%rdx,8),%rdi
      switch (w) {
    496f:	41 83 fd 10          	cmp    $0x10,%r13d
    4973:	75 8a                	jne    48ff <jerasure_matrix_dotprod+0x14f>
        case 16: galois_w16_region_multiply(sptr, matrix_row[i], size, dptr, init); break;
    4975:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    497a:	44 89 f2             	mov    %r14d,%edx
    497d:	e8 ee 31 00 00       	callq  7b70 <galois_w16_region_multiply>
    4982:	eb 8f                	jmp    4913 <jerasure_matrix_dotprod+0x163>
    4984:	0f 1f 40 00          	nopl   0x0(%rax)
    4988:	48 8b 0d b1 c7 00 00 	mov    0xc7b1(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    498f:	48 8d 3d d2 5d 00 00 	lea    0x5dd2(%rip),%rdi        # a768 <__PRETTY_FUNCTION__.5741+0x8f>
    4996:	ba 44 00 00 00       	mov    $0x44,%edx
    499b:	be 01 00 00 00       	mov    $0x1,%esi
    49a0:	e8 cb ca ff ff       	callq  1470 <fwrite@plt>
    exit(1);
    49a5:	bf 01 00 00 00       	mov    $0x1,%edi
    49aa:	e8 b1 ca ff ff       	callq  1460 <exit@plt>
    49af:	90                   	nop
  dptr = (dest_id < k) ? data_ptrs[dest_id] : coding_ptrs[dest_id-k];
    49b0:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    49b5:	4d 63 c0             	movslq %r8d,%r8
    49b8:	4e 8b 34 c0          	mov    (%rax,%r8,8),%r14
  for (i = 0; i < k; i++) {
    49bc:	8b 44 24 18          	mov    0x18(%rsp),%eax
    49c0:	85 c0                	test   %eax,%eax
    49c2:	0f 8f 55 fe ff ff    	jg     481d <jerasure_matrix_dotprod+0x6d>
}
    49c8:	48 83 c4 28          	add    $0x28,%rsp
    49cc:	5b                   	pop    %rbx
    49cd:	5d                   	pop    %rbp
    49ce:	41 5c                	pop    %r12
    49d0:	41 5d                	pop    %r13
    49d2:	41 5e                	pop    %r14
    49d4:	41 5f                	pop    %r15
    49d6:	c3                   	retq   
    49d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    49de:	00 00 
        galois_region_xor(sptr, dptr, dptr, size);
    49e0:	44 89 c9             	mov    %r9d,%ecx
    49e3:	48 89 ea             	mov    %rbp,%rdx
    49e6:	48 89 ee             	mov    %rbp,%rsi
    49e9:	44 89 4c 24 68       	mov    %r9d,0x68(%rsp)
    49ee:	e8 2d 37 00 00       	callq  8120 <galois_region_xor>
        jerasure_total_xor_bytes += size;
    49f3:	44 8b 4c 24 68       	mov    0x68(%rsp),%r9d
    49f8:	c5 e1 57 db          	vxorpd %xmm3,%xmm3,%xmm3
    49fc:	c4 c1 63 2a c1       	vcvtsi2sd %r9d,%xmm3,%xmm0
    4a01:	c5 fb 58 05 57 c7 00 	vaddsd 0xc757(%rip),%xmm0,%xmm0        # 11160 <jerasure_total_xor_bytes>
    4a08:	00 
    4a09:	c5 fb 11 05 4f c7 00 	vmovsd %xmm0,0xc74f(%rip)        # 11160 <jerasure_total_xor_bytes>
    4a10:	00 
    4a11:	e9 2a fe ff ff       	jmpq   4840 <jerasure_matrix_dotprod+0x90>
    4a16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    4a1d:	00 00 00 
        sptr = coding_ptrs[src_ids[i]-k];
    4a20:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    4a25:	2b 54 24 18          	sub    0x18(%rsp),%edx
    4a29:	48 63 d2             	movslq %edx,%rdx
    4a2c:	48 8b 3c d0          	mov    (%rax,%rdx,8),%rdi
    4a30:	e9 3f fe ff ff       	jmpq   4874 <jerasure_matrix_dotprod+0xc4>
    4a35:	0f 1f 00             	nopl   (%rax)
        sptr = data_ptrs[i];
    4a38:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    4a3d:	4a 8b 3c f8          	mov    (%rax,%r15,8),%rdi
    4a41:	e9 b3 fe ff ff       	jmpq   48f9 <jerasure_matrix_dotprod+0x149>
    4a46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    4a4d:	00 00 00 
        case 8:  galois_w08_region_multiply(sptr, matrix_row[i], size, dptr, init); break;
    4a50:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    4a55:	44 89 f2             	mov    %r14d,%edx
    4a58:	e8 93 2f 00 00       	callq  79f0 <galois_w08_region_multiply>
    4a5d:	e9 b1 fe ff ff       	jmpq   4913 <jerasure_matrix_dotprod+0x163>
    4a62:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
        case 32: galois_w32_region_multiply(sptr, matrix_row[i], size, dptr, init); break;
    4a68:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    4a6d:	44 89 f2             	mov    %r14d,%edx
    4a70:	e8 2b 38 00 00       	callq  82a0 <galois_w32_region_multiply>
    4a75:	e9 99 fe ff ff       	jmpq   4913 <jerasure_matrix_dotprod+0x163>
    4a7a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
        sptr = data_ptrs[i];
    4a80:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    4a85:	4a 8b 3c f8          	mov    (%rax,%r15,8),%rdi
    4a89:	e9 e6 fd ff ff       	jmpq   4874 <jerasure_matrix_dotprod+0xc4>
    4a8e:	66 90                	xchg   %ax,%ax

0000000000004a90 <jerasure_matrix_decode>:
{
    4a90:	f3 0f 1e fa          	endbr64 
    4a94:	41 57                	push   %r15
    4a96:	41 56                	push   %r14
    4a98:	41 55                	push   %r13
    4a9a:	41 89 d5             	mov    %edx,%r13d
  if (w != 8 && w != 16 && w != 32) return -1;
    4a9d:	41 8d 45 f8          	lea    -0x8(%r13),%eax
{
    4aa1:	41 54                	push   %r12
    4aa3:	4c 89 ca             	mov    %r9,%rdx
    4aa6:	55                   	push   %rbp
    4aa7:	44 89 c5             	mov    %r8d,%ebp
    4aaa:	53                   	push   %rbx
    4aab:	89 fb                	mov    %edi,%ebx
    4aad:	48 83 ec 48          	sub    $0x48,%rsp
  if (w != 8 && w != 16 && w != 32) return -1;
    4ab1:	83 e0 f7             	and    $0xfffffff7,%eax
{
    4ab4:	89 74 24 20          	mov    %esi,0x20(%rsp)
    4ab8:	48 89 4c 24 18       	mov    %rcx,0x18(%rsp)
  if (w != 8 && w != 16 && w != 32) return -1;
    4abd:	74 0a                	je     4ac9 <jerasure_matrix_decode+0x39>
    4abf:	41 83 fd 20          	cmp    $0x20,%r13d
    4ac3:	0f 85 e9 03 00 00    	jne    4eb2 <jerasure_matrix_decode+0x422>
  erased = jerasure_erasures_to_erased(k, m, erasures);
    4ac9:	8b 74 24 20          	mov    0x20(%rsp),%esi
    4acd:	89 df                	mov    %ebx,%edi
    4acf:	e8 fc f9 ff ff       	callq  44d0 <jerasure_erasures_to_erased>
    4ad4:	49 89 c6             	mov    %rax,%r14
  if (erased == NULL) return -1;
    4ad7:	48 85 c0             	test   %rax,%rax
    4ada:	0f 84 d2 03 00 00    	je     4eb2 <jerasure_matrix_decode+0x422>
  for (i = 0; i < k; i++) {
    4ae0:	85 db                	test   %ebx,%ebx
    4ae2:	0f 8e 68 03 00 00    	jle    4e50 <jerasure_matrix_decode+0x3c0>
    4ae8:	8d 4b ff             	lea    -0x1(%rbx),%ecx
    4aeb:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
    4aef:	41 89 df             	mov    %ebx,%r15d
    4af2:	31 c0                	xor    %eax,%eax
  edd = 0;
    4af4:	45 31 e4             	xor    %r12d,%r12d
    4af7:	eb 0a                	jmp    4b03 <jerasure_matrix_decode+0x73>
    4af9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    4b00:	48 89 d0             	mov    %rdx,%rax
    if (erased[i]) {
    4b03:	45 8b 04 86          	mov    (%r14,%rax,4),%r8d
    4b07:	45 85 c0             	test   %r8d,%r8d
    4b0a:	74 06                	je     4b12 <jerasure_matrix_decode+0x82>
      edd++;
    4b0c:	41 ff c4             	inc    %r12d
    4b0f:	41 89 c7             	mov    %eax,%r15d
  for (i = 0; i < k; i++) {
    4b12:	48 8d 50 01          	lea    0x1(%rax),%rdx
    4b16:	48 39 c1             	cmp    %rax,%rcx
    4b19:	75 e5                	jne    4b00 <jerasure_matrix_decode+0x70>
  if (!row_k_ones || erased[k]) lastdrive = k;
    4b1b:	85 ed                	test   %ebp,%ebp
    4b1d:	0f 84 2d 02 00 00    	je     4d50 <jerasure_matrix_decode+0x2c0>
    4b23:	48 63 c3             	movslq %ebx,%rax
    4b26:	41 8b 3c 86          	mov    (%r14,%rax,4),%edi
    4b2a:	85 ff                	test   %edi,%edi
    4b2c:	44 0f 45 fb          	cmovne %ebx,%r15d
  if (edd > 1 || (edd > 0 && (!row_k_ones || erased[k]))) {
    4b30:	41 83 fc 01          	cmp    $0x1,%r12d
    4b34:	0f 8f 23 02 00 00    	jg     4d5d <jerasure_matrix_decode+0x2cd>
    4b3a:	0f 85 f8 02 00 00    	jne    4e38 <jerasure_matrix_decode+0x3a8>
    4b40:	48 63 c3             	movslq %ebx,%rax
    4b43:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    4b48:	48 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%rdi
    4b4f:	00 
    4b50:	85 ed                	test   %ebp,%ebp
    4b52:	0f 85 98 02 00 00    	jne    4df0 <jerasure_matrix_decode+0x360>
    dm_ids = talloc(int, k);
    4b58:	e8 a3 c8 ff ff       	callq  1400 <malloc@plt>
    4b5d:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    if (dm_ids == NULL) {
    4b62:	48 85 c0             	test   %rax,%rax
    4b65:	0f 84 3f 03 00 00    	je     4eaa <jerasure_matrix_decode+0x41a>
    decoding_matrix = talloc(int, k*k);
    4b6b:	89 df                	mov    %ebx,%edi
    4b6d:	0f af fb             	imul   %ebx,%edi
    4b70:	48 63 ff             	movslq %edi,%rdi
    4b73:	48 c1 e7 02          	shl    $0x2,%rdi
    4b77:	e8 84 c8 ff ff       	callq  1400 <malloc@plt>
    4b7c:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    if (decoding_matrix == NULL) {
    4b81:	48 85 c0             	test   %rax,%rax
    4b84:	0f 84 5a 03 00 00    	je     4ee4 <jerasure_matrix_decode+0x454>
    if (jerasure_make_decoding_matrix(k, m, w, matrix, erased, decoding_matrix, dm_ids) < 0) {
    4b8a:	48 83 ec 08          	sub    $0x8,%rsp
    4b8e:	ff 74 24 10          	pushq  0x10(%rsp)
    4b92:	44 89 ea             	mov    %r13d,%edx
    4b95:	4d 89 f0             	mov    %r14,%r8
    4b98:	4c 8b 4c 24 20       	mov    0x20(%rsp),%r9
    4b9d:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
    4ba2:	8b 74 24 30          	mov    0x30(%rsp),%esi
    4ba6:	89 df                	mov    %ebx,%edi
    4ba8:	e8 03 f5 ff ff       	callq  40b0 <jerasure_make_decoding_matrix>
    4bad:	5a                   	pop    %rdx
    4bae:	59                   	pop    %rcx
    4baf:	85 c0                	test   %eax,%eax
    4bb1:	0f 88 06 03 00 00    	js     4ebd <jerasure_matrix_decode+0x42d>
  for (i = 0; edd > 0 && i < lastdrive; i++) {
    4bb7:	45 85 ff             	test   %r15d,%r15d
    4bba:	0f 8e b8 01 00 00    	jle    4d78 <jerasure_matrix_decode+0x2e8>
    4bc0:	4c 89 f5             	mov    %r14,%rbp
  decoding_matrix = NULL;
    4bc3:	31 c0                	xor    %eax,%eax
  for (i = 0; edd > 0 && i < lastdrive; i++) {
    4bc5:	45 31 c0             	xor    %r8d,%r8d
    4bc8:	4c 89 74 24 30       	mov    %r14,0x30(%rsp)
    4bcd:	44 89 6c 24 38       	mov    %r13d,0x38(%rsp)
    4bd2:	41 89 de             	mov    %ebx,%r14d
    4bd5:	45 89 fd             	mov    %r15d,%r13d
    4bd8:	48 89 eb             	mov    %rbp,%rbx
    4bdb:	45 89 e7             	mov    %r12d,%r15d
    4bde:	44 89 c5             	mov    %r8d,%ebp
    4be1:	41 89 c4             	mov    %eax,%r12d
    4be4:	eb 1d                	jmp    4c03 <jerasure_matrix_decode+0x173>
    4be6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    4bed:	00 00 00 
    4bf0:	ff c5                	inc    %ebp
    4bf2:	48 83 c3 04          	add    $0x4,%rbx
    4bf6:	45 01 f4             	add    %r14d,%r12d
    4bf9:	45 85 ff             	test   %r15d,%r15d
    4bfc:	7e 5a                	jle    4c58 <jerasure_matrix_decode+0x1c8>
    4bfe:	41 39 ed             	cmp    %ebp,%r13d
    4c01:	7e 55                	jle    4c58 <jerasure_matrix_decode+0x1c8>
    if (erased[i]) {
    4c03:	8b 03                	mov    (%rbx),%eax
    4c05:	85 c0                	test   %eax,%eax
    4c07:	74 e7                	je     4bf0 <jerasure_matrix_decode+0x160>
      jerasure_matrix_dotprod(k, w, decoding_matrix+(i*k), dm_ids, i, data_ptrs, coding_ptrs, size);
    4c09:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    4c0e:	49 63 d4             	movslq %r12d,%rdx
    4c11:	48 8d 14 90          	lea    (%rax,%rdx,4),%rdx
    4c15:	8b 84 24 90 00 00 00 	mov    0x90(%rsp),%eax
    4c1c:	41 89 e8             	mov    %ebp,%r8d
    4c1f:	50                   	push   %rax
    4c20:	44 89 f7             	mov    %r14d,%edi
      edd--;
    4c23:	41 ff cf             	dec    %r15d
      jerasure_matrix_dotprod(k, w, decoding_matrix+(i*k), dm_ids, i, data_ptrs, coding_ptrs, size);
    4c26:	ff b4 24 90 00 00 00 	pushq  0x90(%rsp)
  for (i = 0; edd > 0 && i < lastdrive; i++) {
    4c2d:	ff c5                	inc    %ebp
    4c2f:	48 83 c3 04          	add    $0x4,%rbx
      jerasure_matrix_dotprod(k, w, decoding_matrix+(i*k), dm_ids, i, data_ptrs, coding_ptrs, size);
    4c33:	4c 8b 8c 24 90 00 00 	mov    0x90(%rsp),%r9
    4c3a:	00 
    4c3b:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
    4c40:	8b 74 24 48          	mov    0x48(%rsp),%esi
    4c44:	45 01 f4             	add    %r14d,%r12d
    4c47:	e8 64 fb ff ff       	callq  47b0 <jerasure_matrix_dotprod>
      edd--;
    4c4c:	41 5a                	pop    %r10
    4c4e:	41 5b                	pop    %r11
  for (i = 0; edd > 0 && i < lastdrive; i++) {
    4c50:	45 85 ff             	test   %r15d,%r15d
    4c53:	7f a9                	jg     4bfe <jerasure_matrix_decode+0x16e>
    4c55:	0f 1f 00             	nopl   (%rax)
    4c58:	45 89 fc             	mov    %r15d,%r12d
    4c5b:	44 89 f3             	mov    %r14d,%ebx
    4c5e:	45 89 ef             	mov    %r13d,%r15d
    4c61:	4c 8b 74 24 30       	mov    0x30(%rsp),%r14
    4c66:	44 8b 6c 24 38       	mov    0x38(%rsp),%r13d
  if (edd > 0) {
    4c6b:	45 85 e4             	test   %r12d,%r12d
    4c6e:	0f 85 04 01 00 00    	jne    4d78 <jerasure_matrix_decode+0x2e8>
  for (i = 0; i < m; i++) {
    4c74:	8b 7c 24 20          	mov    0x20(%rsp),%edi
    4c78:	48 63 c3             	movslq %ebx,%rax
    4c7b:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    4c80:	85 ff                	test   %edi,%edi
    4c82:	0f 8e 86 00 00 00    	jle    4d0e <jerasure_matrix_decode+0x27e>
    4c88:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  decoding_matrix = NULL;
    4c8d:	41 89 df             	mov    %ebx,%r15d
    4c90:	49 8d 2c 86          	lea    (%r14,%rax,4),%rbp
    4c94:	8b 44 24 20          	mov    0x20(%rsp),%eax
    4c98:	4c 89 74 24 20       	mov    %r14,0x20(%rsp)
    4c9d:	01 d8                	add    %ebx,%eax
    4c9f:	41 89 c6             	mov    %eax,%r14d
    4ca2:	44 89 e8             	mov    %r13d,%eax
    4ca5:	45 31 e4             	xor    %r12d,%r12d
    4ca8:	41 89 dd             	mov    %ebx,%r13d
    4cab:	89 c3                	mov    %eax,%ebx
    4cad:	eb 10                	jmp    4cbf <jerasure_matrix_decode+0x22f>
    4caf:	90                   	nop
  for (i = 0; i < m; i++) {
    4cb0:	41 ff c7             	inc    %r15d
    4cb3:	48 83 c5 04          	add    $0x4,%rbp
    4cb7:	45 01 ec             	add    %r13d,%r12d
    4cba:	45 39 fe             	cmp    %r15d,%r14d
    4cbd:	74 4a                	je     4d09 <jerasure_matrix_decode+0x279>
    if (erased[k+i]) {
    4cbf:	8b 75 00             	mov    0x0(%rbp),%esi
    4cc2:	85 f6                	test   %esi,%esi
    4cc4:	74 ea                	je     4cb0 <jerasure_matrix_decode+0x220>
      jerasure_matrix_dotprod(k, w, matrix+(i*k), NULL, i+k, data_ptrs, coding_ptrs, size);
    4cc6:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    4ccb:	49 63 d4             	movslq %r12d,%rdx
    4cce:	48 8d 14 90          	lea    (%rax,%rdx,4),%rdx
    4cd2:	8b 84 24 90 00 00 00 	mov    0x90(%rsp),%eax
    4cd9:	45 89 f8             	mov    %r15d,%r8d
    4cdc:	50                   	push   %rax
    4cdd:	31 c9                	xor    %ecx,%ecx
    4cdf:	89 de                	mov    %ebx,%esi
    4ce1:	ff b4 24 90 00 00 00 	pushq  0x90(%rsp)
    4ce8:	44 89 ef             	mov    %r13d,%edi
    4ceb:	41 ff c7             	inc    %r15d
    4cee:	4c 8b 8c 24 90 00 00 	mov    0x90(%rsp),%r9
    4cf5:	00 
    4cf6:	48 83 c5 04          	add    $0x4,%rbp
    4cfa:	e8 b1 fa ff ff       	callq  47b0 <jerasure_matrix_dotprod>
    4cff:	5a                   	pop    %rdx
    4d00:	59                   	pop    %rcx
  for (i = 0; i < m; i++) {
    4d01:	45 01 ec             	add    %r13d,%r12d
    4d04:	45 39 fe             	cmp    %r15d,%r14d
    4d07:	75 b6                	jne    4cbf <jerasure_matrix_decode+0x22f>
    4d09:	4c 8b 74 24 20       	mov    0x20(%rsp),%r14
  free(erased);
    4d0e:	4c 89 f7             	mov    %r14,%rdi
    4d11:	e8 6a c5 ff ff       	callq  1280 <free@plt>
  if (dm_ids != NULL) free(dm_ids);
    4d16:	48 83 7c 24 08 00    	cmpq   $0x0,0x8(%rsp)
    4d1c:	74 0a                	je     4d28 <jerasure_matrix_decode+0x298>
    4d1e:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    4d23:	e8 58 c5 ff ff       	callq  1280 <free@plt>
  if (decoding_matrix != NULL) free(decoding_matrix);
    4d28:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  return 0;
    4d2d:	45 31 e4             	xor    %r12d,%r12d
  if (decoding_matrix != NULL) free(decoding_matrix);
    4d30:	48 85 c0             	test   %rax,%rax
    4d33:	74 08                	je     4d3d <jerasure_matrix_decode+0x2ad>
    4d35:	48 89 c7             	mov    %rax,%rdi
    4d38:	e8 43 c5 ff ff       	callq  1280 <free@plt>
}
    4d3d:	48 83 c4 48          	add    $0x48,%rsp
    4d41:	5b                   	pop    %rbx
    4d42:	5d                   	pop    %rbp
    4d43:	44 89 e0             	mov    %r12d,%eax
    4d46:	41 5c                	pop    %r12
    4d48:	41 5d                	pop    %r13
    4d4a:	41 5e                	pop    %r14
    4d4c:	41 5f                	pop    %r15
    4d4e:	c3                   	retq   
    4d4f:	90                   	nop
    4d50:	41 89 df             	mov    %ebx,%r15d
  if (edd > 1 || (edd > 0 && (!row_k_ones || erased[k]))) {
    4d53:	41 83 fc 01          	cmp    $0x1,%r12d
    4d57:	0f 8e dd fd ff ff    	jle    4b3a <jerasure_matrix_decode+0xaa>
    4d5d:	48 63 c3             	movslq %ebx,%rax
    4d60:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    4d65:	48 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%rdi
    4d6c:	00 
    4d6d:	e9 e6 fd ff ff       	jmpq   4b58 <jerasure_matrix_decode+0xc8>
    4d72:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    tmpids = talloc(int, k);
    4d78:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    4d7d:	48 c1 e7 02          	shl    $0x2,%rdi
    4d81:	e8 7a c6 ff ff       	callq  1400 <malloc@plt>
    4d86:	48 89 c5             	mov    %rax,%rbp
    for (i = 0; i < k; i++) {
    4d89:	85 db                	test   %ebx,%ebx
    4d8b:	7e 24                	jle    4db1 <jerasure_matrix_decode+0x321>
    4d8d:	8b 4c 24 3c          	mov    0x3c(%rsp),%ecx
  decoding_matrix = NULL;
    4d91:	31 d2                	xor    %edx,%edx
    4d93:	eb 06                	jmp    4d9b <jerasure_matrix_decode+0x30b>
    4d95:	0f 1f 00             	nopl   (%rax)
    4d98:	48 89 c2             	mov    %rax,%rdx
      tmpids[i] = (i < lastdrive) ? i : i+1;
    4d9b:	8d 42 01             	lea    0x1(%rdx),%eax
    4d9e:	41 39 d7             	cmp    %edx,%r15d
    4da1:	0f 4f c2             	cmovg  %edx,%eax
    4da4:	89 44 95 00          	mov    %eax,0x0(%rbp,%rdx,4)
    for (i = 0; i < k; i++) {
    4da8:	48 8d 42 01          	lea    0x1(%rdx),%rax
    4dac:	48 39 ca             	cmp    %rcx,%rdx
    4daf:	75 e7                	jne    4d98 <jerasure_matrix_decode+0x308>
    jerasure_matrix_dotprod(k, w, matrix, tmpids, lastdrive, data_ptrs, coding_ptrs, size);
    4db1:	8b 84 24 90 00 00 00 	mov    0x90(%rsp),%eax
    4db8:	45 89 f8             	mov    %r15d,%r8d
    4dbb:	50                   	push   %rax
    4dbc:	48 89 e9             	mov    %rbp,%rcx
    4dbf:	44 89 ee             	mov    %r13d,%esi
    4dc2:	ff b4 24 90 00 00 00 	pushq  0x90(%rsp)
    4dc9:	89 df                	mov    %ebx,%edi
    4dcb:	4c 8b 8c 24 90 00 00 	mov    0x90(%rsp),%r9
    4dd2:	00 
    4dd3:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    4dd8:	e8 d3 f9 ff ff       	callq  47b0 <jerasure_matrix_dotprod>
    free(tmpids);
    4ddd:	48 89 ef             	mov    %rbp,%rdi
    4de0:	e8 9b c4 ff ff       	callq  1280 <free@plt>
    4de5:	41 58                	pop    %r8
    4de7:	41 59                	pop    %r9
    4de9:	e9 86 fe ff ff       	jmpq   4c74 <jerasure_matrix_decode+0x1e4>
    4dee:	66 90                	xchg   %ax,%ax
  if (edd > 1 || (edd > 0 && (!row_k_ones || erased[k]))) {
    4df0:	41 8b 34 86          	mov    (%r14,%rax,4),%esi
    4df4:	85 f6                	test   %esi,%esi
    4df6:	0f 85 5c fd ff ff    	jne    4b58 <jerasure_matrix_decode+0xc8>
  dm_ids = NULL;
    4dfc:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    4e03:	00 00 
  decoding_matrix = NULL;
    4e05:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    4e0c:	00 00 
  for (i = 0; edd > 0 && i < lastdrive; i++) {
    4e0e:	45 85 ff             	test   %r15d,%r15d
    4e11:	0f 85 a9 fd ff ff    	jne    4bc0 <jerasure_matrix_decode+0x130>
    tmpids = talloc(int, k);
    4e17:	e8 e4 c5 ff ff       	callq  1400 <malloc@plt>
    4e1c:	48 89 c5             	mov    %rax,%rbp
    for (i = 0; i < k; i++) {
    4e1f:	e9 69 ff ff ff       	jmpq   4d8d <jerasure_matrix_decode+0x2fd>
  if (!row_k_ones || erased[k]) lastdrive = k;
    4e24:	48 63 c3             	movslq %ebx,%rax
    4e27:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    4e2c:	41 8b 04 86          	mov    (%r14,%rax,4),%eax
    4e30:	85 c0                	test   %eax,%eax
    4e32:	74 48                	je     4e7c <jerasure_matrix_decode+0x3ec>
    4e34:	0f 1f 40 00          	nopl   0x0(%rax)
  dm_ids = NULL;
    4e38:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    4e3f:	00 00 
  decoding_matrix = NULL;
    4e41:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    4e48:	00 00 
    4e4a:	e9 25 fe ff ff       	jmpq   4c74 <jerasure_matrix_decode+0x1e4>
    4e4f:	90                   	nop
  if (!row_k_ones || erased[k]) lastdrive = k;
    4e50:	85 ed                	test   %ebp,%ebp
    4e52:	75 d0                	jne    4e24 <jerasure_matrix_decode+0x394>
  for (i = 0; i < m; i++) {
    4e54:	48 63 c3             	movslq %ebx,%rax
    4e57:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    4e5c:	8b 44 24 20          	mov    0x20(%rsp),%eax
  dm_ids = NULL;
    4e60:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    4e67:	00 00 
  decoding_matrix = NULL;
    4e69:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    4e70:	00 00 
  for (i = 0; i < m; i++) {
    4e72:	85 c0                	test   %eax,%eax
    4e74:	0f 8f 0e fe ff ff    	jg     4c88 <jerasure_matrix_decode+0x1f8>
    4e7a:	eb 1e                	jmp    4e9a <jerasure_matrix_decode+0x40a>
    4e7c:	8b 44 24 20          	mov    0x20(%rsp),%eax
  dm_ids = NULL;
    4e80:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    4e87:	00 00 
  decoding_matrix = NULL;
    4e89:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    4e90:	00 00 
  for (i = 0; i < m; i++) {
    4e92:	85 c0                	test   %eax,%eax
    4e94:	0f 8f ee fd ff ff    	jg     4c88 <jerasure_matrix_decode+0x1f8>
  free(erased);
    4e9a:	4c 89 f7             	mov    %r14,%rdi
    4e9d:	e8 de c3 ff ff       	callq  1280 <free@plt>
  return 0;
    4ea2:	45 31 e4             	xor    %r12d,%r12d
    4ea5:	e9 93 fe ff ff       	jmpq   4d3d <jerasure_matrix_decode+0x2ad>
      free(erased);
    4eaa:	4c 89 f7             	mov    %r14,%rdi
    4ead:	e8 ce c3 ff ff       	callq  1280 <free@plt>
      return -1;
    4eb2:	41 bc ff ff ff ff    	mov    $0xffffffff,%r12d
    4eb8:	e9 80 fe ff ff       	jmpq   4d3d <jerasure_matrix_decode+0x2ad>
      free(erased);
    4ebd:	4c 89 f7             	mov    %r14,%rdi
    4ec0:	e8 bb c3 ff ff       	callq  1280 <free@plt>
      free(dm_ids);
    4ec5:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
      return -1;
    4eca:	41 bc ff ff ff ff    	mov    $0xffffffff,%r12d
      free(dm_ids);
    4ed0:	e8 ab c3 ff ff       	callq  1280 <free@plt>
      free(decoding_matrix);
    4ed5:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    4eda:	e8 a1 c3 ff ff       	callq  1280 <free@plt>
      return -1;
    4edf:	e9 59 fe ff ff       	jmpq   4d3d <jerasure_matrix_decode+0x2ad>
      free(erased);
    4ee4:	4c 89 f7             	mov    %r14,%rdi
    4ee7:	e8 94 c3 ff ff       	callq  1280 <free@plt>
      free(dm_ids);
    4eec:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
      return -1;
    4ef1:	41 83 cc ff          	or     $0xffffffff,%r12d
      free(dm_ids);
    4ef5:	e8 86 c3 ff ff       	callq  1280 <free@plt>
      return -1;
    4efa:	e9 3e fe ff ff       	jmpq   4d3d <jerasure_matrix_decode+0x2ad>
    4eff:	90                   	nop

0000000000004f00 <jerasure_matrix_encode>:
{
    4f00:	f3 0f 1e fa          	endbr64 
    4f04:	41 57                	push   %r15
  if (w != 8 && w != 16 && w != 32) {
    4f06:	8d 42 f8             	lea    -0x8(%rdx),%eax
{
    4f09:	41 89 d7             	mov    %edx,%r15d
    4f0c:	41 56                	push   %r14
    4f0e:	41 55                	push   %r13
    4f10:	4d 89 cd             	mov    %r9,%r13
    4f13:	41 54                	push   %r12
    4f15:	4d 89 c4             	mov    %r8,%r12
    4f18:	55                   	push   %rbp
    4f19:	89 fd                	mov    %edi,%ebp
    4f1b:	53                   	push   %rbx
    4f1c:	48 83 ec 18          	sub    $0x18,%rsp
  if (w != 8 && w != 16 && w != 32) {
    4f20:	83 e0 f7             	and    $0xfffffff7,%eax
    4f23:	74 05                	je     4f2a <jerasure_matrix_encode+0x2a>
    4f25:	83 fa 20             	cmp    $0x20,%edx
    4f28:	75 60                	jne    4f8a <jerasure_matrix_encode+0x8a>
  for (i = 0; i < m; i++) {
    4f2a:	85 f6                	test   %esi,%esi
    4f2c:	7e 4d                	jle    4f7b <jerasure_matrix_encode+0x7b>
    4f2e:	4c 63 f5             	movslq %ebp,%r14
    4f31:	4a 8d 04 b5 00 00 00 	lea    0x0(,%r14,4),%rax
    4f38:	00 
    4f39:	48 89 04 24          	mov    %rax,(%rsp)
    4f3d:	8d 04 2e             	lea    (%rsi,%rbp,1),%eax
    4f40:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    4f44:	49 89 ce             	mov    %rcx,%r14
    4f47:	89 eb                	mov    %ebp,%ebx
    4f49:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    jerasure_matrix_dotprod(k, w, matrix+(i*k), NULL, k+i, data_ptrs, coding_ptrs, size);
    4f50:	8b 44 24 50          	mov    0x50(%rsp),%eax
    4f54:	4c 89 f2             	mov    %r14,%rdx
    4f57:	50                   	push   %rax
    4f58:	41 89 d8             	mov    %ebx,%r8d
    4f5b:	4d 89 e1             	mov    %r12,%r9
    4f5e:	41 55                	push   %r13
    4f60:	31 c9                	xor    %ecx,%ecx
    4f62:	44 89 fe             	mov    %r15d,%esi
    4f65:	89 ef                	mov    %ebp,%edi
    4f67:	e8 44 f8 ff ff       	callq  47b0 <jerasure_matrix_dotprod>
  for (i = 0; i < m; i++) {
    4f6c:	4c 03 74 24 10       	add    0x10(%rsp),%r14
    4f71:	58                   	pop    %rax
    4f72:	ff c3                	inc    %ebx
    4f74:	5a                   	pop    %rdx
    4f75:	3b 5c 24 0c          	cmp    0xc(%rsp),%ebx
    4f79:	75 d5                	jne    4f50 <jerasure_matrix_encode+0x50>
}
    4f7b:	48 83 c4 18          	add    $0x18,%rsp
    4f7f:	5b                   	pop    %rbx
    4f80:	5d                   	pop    %rbp
    4f81:	41 5c                	pop    %r12
    4f83:	41 5d                	pop    %r13
    4f85:	41 5e                	pop    %r14
    4f87:	41 5f                	pop    %r15
    4f89:	c3                   	retq   
    4f8a:	48 8b 0d af c1 00 00 	mov    0xc1af(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    4f91:	48 8d 3d 18 58 00 00 	lea    0x5818(%rip),%rdi        # a7b0 <__PRETTY_FUNCTION__.5741+0xd7>
    4f98:	ba 39 00 00 00       	mov    $0x39,%edx
    4f9d:	be 01 00 00 00       	mov    $0x1,%esi
    4fa2:	e8 c9 c4 ff ff       	callq  1470 <fwrite@plt>
    exit(1);
    4fa7:	bf 01 00 00 00       	mov    $0x1,%edi
    4fac:	e8 af c4 ff ff       	callq  1460 <exit@plt>
    4fb1:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    4fb8:	00 00 00 00 
    4fbc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000004fc0 <jerasure_invert_bitmatrix>:
  return scache;

}

int jerasure_invert_bitmatrix(int *mat, int *inv, int rows)
{
    4fc0:	f3 0f 1e fa          	endbr64 
    4fc4:	41 57                	push   %r15
    4fc6:	41 56                	push   %r14
    4fc8:	49 89 f6             	mov    %rsi,%r14
  int tmp;
 
  cols = rows;

  k = 0;
  for (i = 0; i < rows; i++) {
    4fcb:	8d 72 ff             	lea    -0x1(%rdx),%esi
{
    4fce:	41 55                	push   %r13
    4fd0:	41 54                	push   %r12
    4fd2:	55                   	push   %rbp
    4fd3:	53                   	push   %rbx
    4fd4:	89 74 24 f4          	mov    %esi,-0xc(%rsp)
  for (i = 0; i < rows; i++) {
    4fd8:	85 d2                	test   %edx,%edx
    4fda:	0f 8e a1 02 00 00    	jle    5281 <jerasure_invert_bitmatrix+0x2c1>
    4fe0:	49 89 fd             	mov    %rdi,%r13
    4fe3:	41 89 d7             	mov    %edx,%r15d
    4fe6:	31 ff                	xor    %edi,%edi
    4fe8:	45 31 c9             	xor    %r9d,%r9d
    4feb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    for (j = 0; j < cols; j++) {
    4ff0:	48 63 c7             	movslq %edi,%rax
    4ff3:	49 8d 0c 86          	lea    (%r14,%rax,4),%rcx
{
    4ff7:	31 c0                	xor    %eax,%eax
    4ff9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
      inv[k] = (i == j) ? 1 : 0;
    5000:	31 d2                	xor    %edx,%edx
    5002:	41 39 c1             	cmp    %eax,%r9d
    5005:	0f 94 c2             	sete   %dl
    5008:	89 14 81             	mov    %edx,(%rcx,%rax,4)
    for (j = 0; j < cols; j++) {
    500b:	48 89 c2             	mov    %rax,%rdx
    500e:	48 ff c0             	inc    %rax
    5011:	48 39 f2             	cmp    %rsi,%rdx
    5014:	75 ea                	jne    5000 <jerasure_invert_bitmatrix+0x40>
  for (i = 0; i < rows; i++) {
    5016:	41 8d 41 01          	lea    0x1(%r9),%eax
    501a:	44 01 ff             	add    %r15d,%edi
    501d:	41 39 c7             	cmp    %eax,%r15d
    5020:	74 05                	je     5027 <jerasure_invert_bitmatrix+0x67>
    5022:	41 89 c1             	mov    %eax,%r9d
    5025:	eb c9                	jmp    4ff0 <jerasure_invert_bitmatrix+0x30>
    5027:	4d 63 df             	movslq %r15d,%r11
    502a:	44 89 7c 24 e0       	mov    %r15d,-0x20(%rsp)
    502f:	4d 89 ef             	mov    %r13,%r15
    5032:	89 44 24 f0          	mov    %eax,-0x10(%rsp)
  for (i = 0; i < cols; i++) {

    /* Swap rows if we have a zero i,i element.  If we can't swap, then the 
       matrix was not invertible */

    if ((mat[i*cols+i]) == 0) { 
    5036:	41 8b 3f             	mov    (%r15),%edi
    5039:	4a 8d 04 9d 04 00 00 	lea    0x4(,%r11,4),%rax
    5040:	00 
    5041:	49 8d 5d 04          	lea    0x4(%r13),%rbx
    5045:	48 89 44 24 e8       	mov    %rax,-0x18(%rsp)
  for (i = 0; i < rows; i++) {
    504a:	31 ed                	xor    %ebp,%ebp
    if ((mat[i*cols+i]) == 0) { 
    504c:	44 89 c8             	mov    %r9d,%eax
    504f:	48 89 5c 24 f8       	mov    %rbx,-0x8(%rsp)
    5054:	4c 8d 24 83          	lea    (%rbx,%rax,4),%r12
    5058:	48 89 44 24 d0       	mov    %rax,-0x30(%rsp)
  for (i = 0; i < rows; i++) {
    505d:	c7 44 24 e4 00 00 00 	movl   $0x0,-0x1c(%rsp)
    5064:	00 
    5065:	4e 8d 14 9d 00 00 00 	lea    0x0(,%r11,4),%r10
    506c:	00 
    506d:	4c 89 eb             	mov    %r13,%rbx
    5070:	31 f6                	xor    %esi,%esi
    5072:	44 8d 45 01          	lea    0x1(%rbp),%r8d
    if ((mat[i*cols+i]) == 0) { 
    5076:	85 ff                	test   %edi,%edi
    5078:	0f 84 b1 00 00 00    	je     512f <jerasure_invert_bitmatrix+0x16f>
    507e:	66 90                	xchg   %ax,%ax
        tmp = inv[i*cols+k]; inv[i*cols+k] = inv[j*cols+k]; inv[j*cols+k] = tmp;
      }
    }
 
    /* Now for each j>i, add A_ji*Ai to Aj */
    for (j = i+1; j != rows; j++) {
    5080:	48 3b 6c 24 d0       	cmp    -0x30(%rsp),%rbp
    5085:	0f 84 46 01 00 00    	je     51d1 <jerasure_invert_bitmatrix+0x211>
    508b:	8b 4c 24 e0          	mov    -0x20(%rsp),%ecx
    508f:	49 8d 04 33          	lea    (%r11,%rsi,1),%rax
    5093:	01 4c 24 e4          	add    %ecx,-0x1c(%rsp)
    5097:	4d 01 d4             	add    %r10,%r12
    509a:	4c 89 d9             	mov    %r11,%rcx
    509d:	49 8d 3c b2          	lea    (%r10,%rsi,4),%rdi
    50a1:	48 89 44 24 d8       	mov    %rax,-0x28(%rsp)
    50a6:	48 f7 d9             	neg    %rcx
    50a9:	4c 89 e6             	mov    %r12,%rsi
    50ac:	eb 17                	jmp    50c5 <jerasure_invert_bitmatrix+0x105>
    50ae:	66 90                	xchg   %ax,%ax
    50b0:	41 8d 40 01          	lea    0x1(%r8),%eax
    50b4:	4c 29 d9             	sub    %r11,%rcx
    50b7:	4c 01 d6             	add    %r10,%rsi
    50ba:	4c 01 d7             	add    %r10,%rdi
    50bd:	45 39 c8             	cmp    %r9d,%r8d
    50c0:	74 3e                	je     5100 <jerasure_invert_bitmatrix+0x140>
    50c2:	41 89 c0             	mov    %eax,%r8d
      if (mat[j*cols+i] != 0) {
    50c5:	8b 14 3b             	mov    (%rbx,%rdi,1),%edx
    50c8:	85 d2                	test   %edx,%edx
    50ca:	74 e4                	je     50b0 <jerasure_invert_bitmatrix+0xf0>
    50cc:	48 89 7c 24 c8       	mov    %rdi,-0x38(%rsp)
    50d1:	49 8d 44 3d 00       	lea    0x0(%r13,%rdi,1),%rax
    50d6:	49 8d 14 3e          	lea    (%r14,%rdi,1),%rdx
    50da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
        for (k = 0; k < cols; k++) {
          mat[j*cols+k] ^= mat[i*cols+k]; 
    50e0:	8b 3c 88             	mov    (%rax,%rcx,4),%edi
    50e3:	31 38                	xor    %edi,(%rax)
          inv[j*cols+k] ^= inv[i*cols+k];
    50e5:	48 83 c0 04          	add    $0x4,%rax
    50e9:	8b 3c 8a             	mov    (%rdx,%rcx,4),%edi
    50ec:	31 3a                	xor    %edi,(%rdx)
        for (k = 0; k < cols; k++) {
    50ee:	48 83 c2 04          	add    $0x4,%rdx
    50f2:	48 39 f0             	cmp    %rsi,%rax
    50f5:	75 e9                	jne    50e0 <jerasure_invert_bitmatrix+0x120>
    50f7:	48 8b 7c 24 c8       	mov    -0x38(%rsp),%rdi
    50fc:	eb b2                	jmp    50b0 <jerasure_invert_bitmatrix+0xf0>
    50fe:	66 90                	xchg   %ax,%ax
  for (i = 0; i < cols; i++) {
    5100:	48 8d 45 01          	lea    0x1(%rbp),%rax
    5104:	4c 03 7c 24 e8       	add    -0x18(%rsp),%r15
    5109:	48 83 c3 04          	add    $0x4,%rbx
    510d:	48 3b 6c 24 d0       	cmp    -0x30(%rsp),%rbp
    5112:	0f 84 b9 00 00 00    	je     51d1 <jerasure_invert_bitmatrix+0x211>
    if ((mat[i*cols+i]) == 0) { 
    5118:	41 8b 3f             	mov    (%r15),%edi
    511b:	48 89 c5             	mov    %rax,%rbp
    511e:	48 8b 74 24 d8       	mov    -0x28(%rsp),%rsi
    5123:	44 8d 45 01          	lea    0x1(%rbp),%r8d
    5127:	85 ff                	test   %edi,%edi
    5129:	0f 85 51 ff ff ff    	jne    5080 <jerasure_invert_bitmatrix+0xc0>
      for (j = i+1; j < rows && (mat[j*cols+i]) == 0; j++) ;
    512f:	44 39 44 24 f0       	cmp    %r8d,-0x10(%rsp)
    5134:	0f 8e 64 01 00 00    	jle    529e <jerasure_invert_bitmatrix+0x2de>
    513a:	8b 44 24 e4          	mov    -0x1c(%rsp),%eax
    513e:	03 44 24 e0          	add    -0x20(%rsp),%eax
    5142:	48 98                	cltq   
    5144:	48 01 e8             	add    %rbp,%rax
    5147:	49 8d 54 85 00       	lea    0x0(%r13,%rax,4),%rdx
    514c:	44 89 c0             	mov    %r8d,%eax
    514f:	eb 18                	jmp    5169 <jerasure_invert_bitmatrix+0x1a9>
    5151:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    5158:	8d 48 01             	lea    0x1(%rax),%ecx
    515b:	4c 01 d2             	add    %r10,%rdx
    515e:	44 39 c8             	cmp    %r9d,%eax
    5161:	0f 84 27 01 00 00    	je     528e <jerasure_invert_bitmatrix+0x2ce>
    5167:	89 c8                	mov    %ecx,%eax
    5169:	8b 0a                	mov    (%rdx),%ecx
    516b:	85 c9                	test   %ecx,%ecx
    516d:	74 e9                	je     5158 <jerasure_invert_bitmatrix+0x198>
      if (j == rows) return -1;
    516f:	8b 4c 24 f0          	mov    -0x10(%rsp),%ecx
    5173:	39 c1                	cmp    %eax,%ecx
    5175:	0f 84 13 01 00 00    	je     528e <jerasure_invert_bitmatrix+0x2ce>
        tmp = mat[i*cols+k]; mat[i*cols+k] = mat[j*cols+k]; mat[j*cols+k] = tmp;
    517b:	0f af c1             	imul   %ecx,%eax
    517e:	48 89 74 24 c8       	mov    %rsi,-0x38(%rsp)
    5183:	48 8d 0c b5 00 00 00 	lea    0x0(,%rsi,4),%rcx
    518a:	00 
    518b:	48 98                	cltq   
    518d:	49 8d 54 0d 00       	lea    0x0(%r13,%rcx,1),%rdx
    5192:	48 29 f0             	sub    %rsi,%rax
    5195:	4c 01 f1             	add    %r14,%rcx
    5198:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    519f:	00 
    51a0:	8b 3a                	mov    (%rdx),%edi
    51a2:	8b 34 82             	mov    (%rdx,%rax,4),%esi
    51a5:	89 32                	mov    %esi,(%rdx)
    51a7:	89 3c 82             	mov    %edi,(%rdx,%rax,4)
        tmp = inv[i*cols+k]; inv[i*cols+k] = inv[j*cols+k]; inv[j*cols+k] = tmp;
    51aa:	48 83 c2 04          	add    $0x4,%rdx
    51ae:	8b 39                	mov    (%rcx),%edi
    51b0:	8b 34 81             	mov    (%rcx,%rax,4),%esi
    51b3:	89 31                	mov    %esi,(%rcx)
    51b5:	89 3c 81             	mov    %edi,(%rcx,%rax,4)
      for (k = 0; k < cols; k++) {
    51b8:	48 83 c1 04          	add    $0x4,%rcx
    51bc:	4c 39 e2             	cmp    %r12,%rdx
    51bf:	75 df                	jne    51a0 <jerasure_invert_bitmatrix+0x1e0>
    51c1:	48 8b 74 24 c8       	mov    -0x38(%rsp),%rsi
    for (j = i+1; j != rows; j++) {
    51c6:	48 3b 6c 24 d0       	cmp    -0x30(%rsp),%rbp
    51cb:	0f 85 ba fe ff ff    	jne    508b <jerasure_invert_bitmatrix+0xcb>
    51d1:	49 63 c1             	movslq %r9d,%rax
    51d4:	44 8b 7c 24 e0       	mov    -0x20(%rsp),%r15d
    }
  }

  /* Now the matrix is upper triangular.  Start at the top and multiply down */

  for (i = rows-1; i >= 0; i--) {
    51d9:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
    51de:	4d 8d 54 85 00       	lea    0x0(%r13,%rax,4),%r10
    51e3:	8b 44 24 f4          	mov    -0xc(%rsp),%eax
    51e7:	44 89 fb             	mov    %r15d,%ebx
    51ea:	41 0f af d9          	imul   %r9d,%ebx
    51ee:	4e 8d 04 9d 00 00 00 	lea    0x0(,%r11,4),%r8
    51f5:	00 
    51f6:	4c 8d 24 86          	lea    (%rsi,%rax,4),%r12
    for (j = 0; j < i; j++) {
    51fa:	45 85 c9             	test   %r9d,%r9d
    51fd:	7e 7d                	jle    527c <jerasure_invert_bitmatrix+0x2bc>
    51ff:	48 63 cb             	movslq %ebx,%rcx
    5202:	4c 89 e6             	mov    %r12,%rsi
    5205:	31 ff                	xor    %edi,%edi
    5207:	31 ed                	xor    %ebp,%ebp
    5209:	eb 15                	jmp    5220 <jerasure_invert_bitmatrix+0x260>
    520b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5210:	ff c5                	inc    %ebp
    5212:	4c 29 d9             	sub    %r11,%rcx
    5215:	4c 01 c6             	add    %r8,%rsi
    5218:	4c 01 c7             	add    %r8,%rdi
    521b:	44 39 cd             	cmp    %r9d,%ebp
    521e:	74 4c                	je     526c <jerasure_invert_bitmatrix+0x2ac>
      if (mat[j*cols+i]) {
    5220:	41 8b 04 3a          	mov    (%r10,%rdi,1),%eax
    5224:	85 c0                	test   %eax,%eax
    5226:	74 e8                	je     5210 <jerasure_invert_bitmatrix+0x250>
        for (k = 0; k < cols; k++) {
    5228:	48 89 7c 24 c8       	mov    %rdi,-0x38(%rsp)
    522d:	49 8d 44 3d 00       	lea    0x0(%r13,%rdi,1),%rax
    5232:	49 8d 14 3e          	lea    (%r14,%rdi,1),%rdx
    5236:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    523d:	00 00 00 
          mat[j*cols+k] ^= mat[i*cols+k]; 
    5240:	8b 3c 88             	mov    (%rax,%rcx,4),%edi
    5243:	31 38                	xor    %edi,(%rax)
          inv[j*cols+k] ^= inv[i*cols+k];
    5245:	48 83 c0 04          	add    $0x4,%rax
    5249:	8b 3c 8a             	mov    (%rdx,%rcx,4),%edi
    524c:	31 3a                	xor    %edi,(%rdx)
        for (k = 0; k < cols; k++) {
    524e:	48 83 c2 04          	add    $0x4,%rdx
    5252:	48 39 c6             	cmp    %rax,%rsi
    5255:	75 e9                	jne    5240 <jerasure_invert_bitmatrix+0x280>
    5257:	48 8b 7c 24 c8       	mov    -0x38(%rsp),%rdi
    for (j = 0; j < i; j++) {
    525c:	ff c5                	inc    %ebp
    525e:	4c 29 d9             	sub    %r11,%rcx
    5261:	4c 01 c6             	add    %r8,%rsi
    5264:	4c 01 c7             	add    %r8,%rdi
    5267:	44 39 cd             	cmp    %r9d,%ebp
    526a:	75 b4                	jne    5220 <jerasure_invert_bitmatrix+0x260>
    526c:	44 8d 4d ff          	lea    -0x1(%rbp),%r9d
    5270:	49 83 ea 04          	sub    $0x4,%r10
    5274:	44 29 fb             	sub    %r15d,%ebx
    5277:	45 85 c9             	test   %r9d,%r9d
    527a:	7f 83                	jg     51ff <jerasure_invert_bitmatrix+0x23f>
  for (i = rows-1; i >= 0; i--) {
    527c:	41 ff c9             	dec    %r9d
    527f:	79 ef                	jns    5270 <jerasure_invert_bitmatrix+0x2b0>
        }
      }
    }
  } 
  return 0;
}
    5281:	5b                   	pop    %rbx
    5282:	5d                   	pop    %rbp
    5283:	41 5c                	pop    %r12
    5285:	41 5d                	pop    %r13
    5287:	41 5e                	pop    %r14
  return 0;
    5289:	31 c0                	xor    %eax,%eax
}
    528b:	41 5f                	pop    %r15
    528d:	c3                   	retq   
    528e:	5b                   	pop    %rbx
    528f:	5d                   	pop    %rbp
    5290:	41 5c                	pop    %r12
    5292:	41 5d                	pop    %r13
    5294:	41 5e                	pop    %r14
      if (j == rows) return -1;
    5296:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
}
    529b:	41 5f                	pop    %r15
    529d:	c3                   	retq   
      for (j = i+1; j < rows && (mat[j*cols+i]) == 0; j++) ;
    529e:	44 89 c0             	mov    %r8d,%eax
    52a1:	e9 c9 fe ff ff       	jmpq   516f <jerasure_invert_bitmatrix+0x1af>
    52a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    52ad:	00 00 00 

00000000000052b0 <jerasure_make_decoding_bitmatrix>:
{
    52b0:	f3 0f 1e fa          	endbr64 
    52b4:	41 57                	push   %r15
    52b6:	89 f8                	mov    %edi,%eax
    52b8:	0f af c2             	imul   %edx,%eax
    52bb:	41 56                	push   %r14
    52bd:	0f af c0             	imul   %eax,%eax
    52c0:	41 55                	push   %r13
    52c2:	41 54                	push   %r12
    52c4:	55                   	push   %rbp
    52c5:	89 fd                	mov    %edi,%ebp
    52c7:	48 63 f8             	movslq %eax,%rdi
    52ca:	53                   	push   %rbx
    52cb:	48 c1 e7 02          	shl    $0x2,%rdi
    52cf:	89 d3                	mov    %edx,%ebx
    52d1:	48 83 ec 18          	sub    $0x18,%rsp
    52d5:	48 89 0c 24          	mov    %rcx,(%rsp)
    52d9:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
    52de:	4c 8b 74 24 50       	mov    0x50(%rsp),%r14
  for (i = 0; j < k; i++) {
    52e3:	85 ed                	test   %ebp,%ebp
    52e5:	0f 8e 39 01 00 00    	jle    5424 <jerasure_make_decoding_bitmatrix+0x174>
    52eb:	4c 89 c1             	mov    %r8,%rcx
    52ee:	31 c0                	xor    %eax,%eax
  j = 0;
    52f0:	31 d2                	xor    %edx,%edx
    52f2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    if (erased[i] == 0) {
    52f8:	8b 34 81             	mov    (%rcx,%rax,4),%esi
    52fb:	85 f6                	test   %esi,%esi
    52fd:	75 09                	jne    5308 <jerasure_make_decoding_bitmatrix+0x58>
      dm_ids[j] = i;
    52ff:	48 63 f2             	movslq %edx,%rsi
    5302:	41 89 04 b6          	mov    %eax,(%r14,%rsi,4)
      j++;
    5306:	ff c2                	inc    %edx
  for (i = 0; j < k; i++) {
    5308:	48 ff c0             	inc    %rax
    530b:	39 ea                	cmp    %ebp,%edx
    530d:	7c e9                	jl     52f8 <jerasure_make_decoding_bitmatrix+0x48>
  tmpmat = talloc(int, k*k*w*w);
    530f:	e8 ec c0 ff ff       	callq  1400 <malloc@plt>
    5314:	49 89 c4             	mov    %rax,%r12
  if (tmpmat == NULL) { return -1; }
    5317:	48 85 c0             	test   %rax,%rax
    531a:	0f 84 11 01 00 00    	je     5431 <jerasure_make_decoding_bitmatrix+0x181>
      for (j = 0; j < k*w*w; j++) {
    5320:	89 d8                	mov    %ebx,%eax
    5322:	0f af c5             	imul   %ebp,%eax
    5325:	41 89 dd             	mov    %ebx,%r13d
    5328:	4d 89 f0             	mov    %r14,%r8
    532b:	44 0f af e8          	imul   %eax,%r13d
    532f:	44 8d 48 01          	lea    0x1(%rax),%r9d
    5333:	8d 4d ff             	lea    -0x1(%rbp),%ecx
    5336:	4d 63 f5             	movslq %r13d,%r14
    5339:	41 8d 55 ff          	lea    -0x1(%r13),%edx
    533d:	4d 63 c9             	movslq %r9d,%r9
    5340:	49 c1 e6 02          	shl    $0x2,%r14
    5344:	4c 89 e6             	mov    %r12,%rsi
    5347:	49 8d 7c 94 04       	lea    0x4(%r12,%rdx,4),%rdi
    534c:	4d 8d 7c 88 04       	lea    0x4(%r8,%rcx,4),%r15
    5351:	49 c1 e1 02          	shl    $0x2,%r9
    5355:	45 31 db             	xor    %r11d,%r11d
    5358:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    535f:	00 
    if (dm_ids[i] < k) {
    5360:	41 8b 08             	mov    (%r8),%ecx
    5363:	39 e9                	cmp    %ebp,%ecx
    5365:	0f 8d 85 00 00 00    	jge    53f0 <jerasure_make_decoding_bitmatrix+0x140>
      for (j = 0; j < k*w*w; j++) tmpmat[index+j] = 0;
    536b:	48 89 f0             	mov    %rsi,%rax
    536e:	45 85 ed             	test   %r13d,%r13d
    5371:	7e 14                	jle    5387 <jerasure_make_decoding_bitmatrix+0xd7>
    5373:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5378:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
    537e:	48 83 c0 04          	add    $0x4,%rax
    5382:	48 39 c7             	cmp    %rax,%rdi
    5385:	75 f1                	jne    5378 <jerasure_make_decoding_bitmatrix+0xc8>
      index = i*k*w*w+dm_ids[i]*w;
    5387:	0f af cb             	imul   %ebx,%ecx
    538a:	44 01 d9             	add    %r11d,%ecx
      for (j = 0; j < w; j++) {
    538d:	85 db                	test   %ebx,%ebx
    538f:	7e 1e                	jle    53af <jerasure_make_decoding_bitmatrix+0xff>
    5391:	48 63 c9             	movslq %ecx,%rcx
    5394:	49 8d 0c 8c          	lea    (%r12,%rcx,4),%rcx
    5398:	31 c0                	xor    %eax,%eax
    539a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    53a0:	ff c0                	inc    %eax
        tmpmat[index] = 1;
    53a2:	c7 01 01 00 00 00    	movl   $0x1,(%rcx)
      for (j = 0; j < w; j++) {
    53a8:	4c 01 c9             	add    %r9,%rcx
    53ab:	39 c3                	cmp    %eax,%ebx
    53ad:	75 f1                	jne    53a0 <jerasure_make_decoding_bitmatrix+0xf0>
  for (i = 0; i < k; i++) {
    53af:	49 83 c0 04          	add    $0x4,%r8
    53b3:	45 01 eb             	add    %r13d,%r11d
    53b6:	4c 01 f6             	add    %r14,%rsi
    53b9:	4c 01 f7             	add    %r14,%rdi
    53bc:	4d 39 c7             	cmp    %r8,%r15
    53bf:	75 9f                	jne    5360 <jerasure_make_decoding_bitmatrix+0xb0>
  i = jerasure_invert_bitmatrix(tmpmat, decoding_matrix, k*w);
    53c1:	89 ea                	mov    %ebp,%edx
    53c3:	0f af d3             	imul   %ebx,%edx
    53c6:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
    53cb:	4c 89 e7             	mov    %r12,%rdi
    53ce:	e8 ed fb ff ff       	callq  4fc0 <jerasure_invert_bitmatrix>
  free(tmpmat);
    53d3:	4c 89 e7             	mov    %r12,%rdi
  i = jerasure_invert_bitmatrix(tmpmat, decoding_matrix, k*w);
    53d6:	41 89 c5             	mov    %eax,%r13d
  free(tmpmat);
    53d9:	e8 a2 be ff ff       	callq  1280 <free@plt>
}
    53de:	48 83 c4 18          	add    $0x18,%rsp
    53e2:	5b                   	pop    %rbx
    53e3:	5d                   	pop    %rbp
    53e4:	41 5c                	pop    %r12
    53e6:	44 89 e8             	mov    %r13d,%eax
    53e9:	41 5d                	pop    %r13
    53eb:	41 5e                	pop    %r14
    53ed:	41 5f                	pop    %r15
    53ef:	c3                   	retq   
      mindex = (dm_ids[i]-k)*k*w*w;
    53f0:	29 e9                	sub    %ebp,%ecx
    53f2:	0f af cd             	imul   %ebp,%ecx
    53f5:	0f af cb             	imul   %ebx,%ecx
    53f8:	0f af cb             	imul   %ebx,%ecx
      for (j = 0; j < k*w*w; j++) {
    53fb:	45 85 ed             	test   %r13d,%r13d
    53fe:	7e af                	jle    53af <jerasure_make_decoding_bitmatrix+0xff>
    5400:	48 8b 04 24          	mov    (%rsp),%rax
    5404:	48 63 c9             	movslq %ecx,%rcx
    5407:	4c 8d 14 88          	lea    (%rax,%rcx,4),%r10
    540b:	31 c0                	xor    %eax,%eax
    540d:	0f 1f 00             	nopl   (%rax)
        tmpmat[index+j] = matrix[mindex+j];
    5410:	41 8b 0c 82          	mov    (%r10,%rax,4),%ecx
    5414:	89 0c 86             	mov    %ecx,(%rsi,%rax,4)
      for (j = 0; j < k*w*w; j++) {
    5417:	48 89 c1             	mov    %rax,%rcx
    541a:	48 ff c0             	inc    %rax
    541d:	48 39 d1             	cmp    %rdx,%rcx
    5420:	75 ee                	jne    5410 <jerasure_make_decoding_bitmatrix+0x160>
    5422:	eb 8b                	jmp    53af <jerasure_make_decoding_bitmatrix+0xff>
  tmpmat = talloc(int, k*k*w*w);
    5424:	e8 d7 bf ff ff       	callq  1400 <malloc@plt>
    5429:	49 89 c4             	mov    %rax,%r12
  if (tmpmat == NULL) { return -1; }
    542c:	48 85 c0             	test   %rax,%rax
    542f:	75 90                	jne    53c1 <jerasure_make_decoding_bitmatrix+0x111>
    5431:	41 83 cd ff          	or     $0xffffffff,%r13d
    5435:	eb a7                	jmp    53de <jerasure_make_decoding_bitmatrix+0x12e>
    5437:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    543e:	00 00 

0000000000005440 <jerasure_bitmatrix_decode>:
{
    5440:	f3 0f 1e fa          	endbr64 
    5444:	41 57                	push   %r15
    5446:	41 56                	push   %r14
    5448:	41 89 d6             	mov    %edx,%r14d
    544b:	4c 89 ca             	mov    %r9,%rdx
    544e:	41 55                	push   %r13
    5450:	41 54                	push   %r12
    5452:	41 89 fc             	mov    %edi,%r12d
    5455:	55                   	push   %rbp
    5456:	53                   	push   %rbx
    5457:	44 89 c3             	mov    %r8d,%ebx
    545a:	48 83 ec 48          	sub    $0x48,%rsp
    545e:	89 74 24 38          	mov    %esi,0x38(%rsp)
    5462:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
  erased = jerasure_erasures_to_erased(k, m, erasures);
    5467:	e8 64 f0 ff ff       	callq  44d0 <jerasure_erasures_to_erased>
  if (erased == NULL) return -1;
    546c:	48 85 c0             	test   %rax,%rax
    546f:	0f 84 6a 04 00 00    	je     58df <jerasure_bitmatrix_decode+0x49f>
    5475:	49 89 c7             	mov    %rax,%r15
  for (i = 0; i < k; i++) {
    5478:	45 85 e4             	test   %r12d,%r12d
    547b:	0f 8e ef 03 00 00    	jle    5870 <jerasure_bitmatrix_decode+0x430>
    5481:	41 8d 4c 24 ff       	lea    -0x1(%r12),%ecx
    5486:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
    548a:	45 89 e2             	mov    %r12d,%r10d
    548d:	31 c0                	xor    %eax,%eax
  edd = 0;
    548f:	31 ed                	xor    %ebp,%ebp
    5491:	eb 08                	jmp    549b <jerasure_bitmatrix_decode+0x5b>
    5493:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5498:	48 89 d0             	mov    %rdx,%rax
    if (erased[i]) {
    549b:	41 8b 14 87          	mov    (%r15,%rax,4),%edx
    549f:	85 d2                	test   %edx,%edx
    54a1:	74 05                	je     54a8 <jerasure_bitmatrix_decode+0x68>
      edd++;
    54a3:	ff c5                	inc    %ebp
    54a5:	41 89 c2             	mov    %eax,%r10d
  for (i = 0; i < k; i++) {
    54a8:	48 8d 50 01          	lea    0x1(%rax),%rdx
    54ac:	48 39 c8             	cmp    %rcx,%rax
    54af:	75 e7                	jne    5498 <jerasure_bitmatrix_decode+0x58>
  if (row_k_ones != 1 || erased[k]) lastdrive = k;
    54b1:	83 fb 01             	cmp    $0x1,%ebx
    54b4:	0f 84 3e 03 00 00    	je     57f8 <jerasure_bitmatrix_decode+0x3b8>
    54ba:	45 89 e2             	mov    %r12d,%r10d
  if (edd > 1 || (edd > 0 && (row_k_ones != 1 || erased[k]))) {
    54bd:	83 fd 01             	cmp    $0x1,%ebp
    54c0:	0f 8f 1a 03 00 00    	jg     57e0 <jerasure_bitmatrix_decode+0x3a0>
    54c6:	0f 85 44 03 00 00    	jne    5810 <jerasure_bitmatrix_decode+0x3d0>
    54cc:	49 63 c4             	movslq %r12d,%rax
    54cf:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    54d4:	48 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%rdi
    54db:	00 
    54dc:	83 fb 01             	cmp    $0x1,%ebx
    54df:	0f 84 4b 03 00 00    	je     5830 <jerasure_bitmatrix_decode+0x3f0>
    54e5:	44 89 54 24 08       	mov    %r10d,0x8(%rsp)
    dm_ids = talloc(int, k);
    54ea:	e8 11 bf ff ff       	callq  1400 <malloc@plt>
    if (dm_ids == NULL) {
    54ef:	48 85 c0             	test   %rax,%rax
    dm_ids = talloc(int, k);
    54f2:	48 89 04 24          	mov    %rax,(%rsp)
    if (dm_ids == NULL) {
    54f6:	44 8b 54 24 08       	mov    0x8(%rsp),%r10d
    54fb:	0f 84 29 04 00 00    	je     592a <jerasure_bitmatrix_decode+0x4ea>
    decoding_matrix = talloc(int, k*k*w*w);
    5501:	44 89 e7             	mov    %r12d,%edi
    5504:	41 0f af fe          	imul   %r14d,%edi
    5508:	44 89 54 24 1c       	mov    %r10d,0x1c(%rsp)
    550d:	0f af ff             	imul   %edi,%edi
    5510:	48 63 ff             	movslq %edi,%rdi
    5513:	48 c1 e7 02          	shl    $0x2,%rdi
    5517:	e8 e4 be ff ff       	callq  1400 <malloc@plt>
    if (decoding_matrix == NULL) {
    551c:	48 85 c0             	test   %rax,%rax
    decoding_matrix = talloc(int, k*k*w*w);
    551f:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    if (decoding_matrix == NULL) {
    5524:	44 8b 54 24 1c       	mov    0x1c(%rsp),%r10d
    5529:	0f 84 e1 03 00 00    	je     5910 <jerasure_bitmatrix_decode+0x4d0>
    552f:	44 89 54 24 1c       	mov    %r10d,0x1c(%rsp)
    if (jerasure_make_decoding_bitmatrix(k, m, w, bitmatrix, erased, decoding_matrix, dm_ids) < 0) {
    5534:	48 83 ec 08          	sub    $0x8,%rsp
    5538:	ff 74 24 08          	pushq  0x8(%rsp)
    553c:	4d 89 f8             	mov    %r15,%r8
    553f:	44 89 f2             	mov    %r14d,%edx
    5542:	4c 8b 4c 24 18       	mov    0x18(%rsp),%r9
    5547:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    554c:	8b 74 24 48          	mov    0x48(%rsp),%esi
    5550:	44 89 e7             	mov    %r12d,%edi
    5553:	e8 58 fd ff ff       	callq  52b0 <jerasure_make_decoding_bitmatrix>
    5558:	41 59                	pop    %r9
    555a:	41 5a                	pop    %r10
    555c:	85 c0                	test   %eax,%eax
    555e:	0f 88 86 03 00 00    	js     58ea <jerasure_bitmatrix_decode+0x4aa>
  for (i = 0; edd > 0 && i < lastdrive; i++) {
    5564:	44 8b 54 24 1c       	mov    0x1c(%rsp),%r10d
    5569:	45 85 d2             	test   %r10d,%r10d
    556c:	0f 8e cb 00 00 00    	jle    563d <jerasure_bitmatrix_decode+0x1fd>
    5572:	44 89 f0             	mov    %r14d,%eax
    5575:	41 0f af c6          	imul   %r14d,%eax
    5579:	4c 89 fb             	mov    %r15,%rbx
  decoding_matrix = NULL;
    557c:	45 31 ed             	xor    %r13d,%r13d
    557f:	41 0f af c4          	imul   %r12d,%eax
  for (i = 0; edd > 0 && i < lastdrive; i++) {
    5583:	45 31 c0             	xor    %r8d,%r8d
    5586:	4c 89 7c 24 30       	mov    %r15,0x30(%rsp)
    558b:	44 89 64 24 1c       	mov    %r12d,0x1c(%rsp)
    5590:	44 89 74 24 20       	mov    %r14d,0x20(%rsp)
    5595:	41 89 ec             	mov    %ebp,%r12d
    5598:	45 89 d7             	mov    %r10d,%r15d
    559b:	44 89 ed             	mov    %r13d,%ebp
    559e:	41 89 c6             	mov    %eax,%r14d
    55a1:	49 89 dd             	mov    %rbx,%r13
    55a4:	44 89 c3             	mov    %r8d,%ebx
    55a7:	eb 1a                	jmp    55c3 <jerasure_bitmatrix_decode+0x183>
    55a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    55b0:	ff c3                	inc    %ebx
    55b2:	49 83 c5 04          	add    $0x4,%r13
    55b6:	44 01 f5             	add    %r14d,%ebp
    55b9:	45 85 e4             	test   %r12d,%r12d
    55bc:	7e 6a                	jle    5628 <jerasure_bitmatrix_decode+0x1e8>
    55be:	41 39 df             	cmp    %ebx,%r15d
    55c1:	7e 65                	jle    5628 <jerasure_bitmatrix_decode+0x1e8>
    if (erased[i]) {
    55c3:	41 8b 75 00          	mov    0x0(%r13),%esi
    55c7:	85 f6                	test   %esi,%esi
    55c9:	74 e5                	je     55b0 <jerasure_bitmatrix_decode+0x170>
      jerasure_bitmatrix_dotprod(k, w, decoding_matrix+i*k*w*w, dm_ids, i, data_ptrs, coding_ptrs, size, packetsize);
    55cb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    55d0:	48 63 d5             	movslq %ebp,%rdx
    55d3:	48 83 ec 08          	sub    $0x8,%rsp
    55d7:	48 8d 14 90          	lea    (%rax,%rdx,4),%rdx
    55db:	8b 84 24 a0 00 00 00 	mov    0xa0(%rsp),%eax
    55e2:	41 89 d8             	mov    %ebx,%r8d
    55e5:	50                   	push   %rax
      edd--;
    55e6:	41 ff cc             	dec    %r12d
  for (i = 0; edd > 0 && i < lastdrive; i++) {
    55e9:	ff c3                	inc    %ebx
      jerasure_bitmatrix_dotprod(k, w, decoding_matrix+i*k*w*w, dm_ids, i, data_ptrs, coding_ptrs, size, packetsize);
    55eb:	8b 84 24 a0 00 00 00 	mov    0xa0(%rsp),%eax
    55f2:	49 83 c5 04          	add    $0x4,%r13
    55f6:	50                   	push   %rax
    55f7:	44 01 f5             	add    %r14d,%ebp
    55fa:	ff b4 24 a0 00 00 00 	pushq  0xa0(%rsp)
    5601:	4c 8b 8c 24 a0 00 00 	mov    0xa0(%rsp),%r9
    5608:	00 
    5609:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    560e:	8b 74 24 40          	mov    0x40(%rsp),%esi
    5612:	8b 7c 24 3c          	mov    0x3c(%rsp),%edi
    5616:	e8 a5 e1 ff ff       	callq  37c0 <jerasure_bitmatrix_dotprod>
      edd--;
    561b:	48 83 c4 20          	add    $0x20,%rsp
  for (i = 0; edd > 0 && i < lastdrive; i++) {
    561f:	45 85 e4             	test   %r12d,%r12d
    5622:	7f 9a                	jg     55be <jerasure_bitmatrix_decode+0x17e>
    5624:	0f 1f 40 00          	nopl   0x0(%rax)
    5628:	44 89 e5             	mov    %r12d,%ebp
    562b:	45 89 fa             	mov    %r15d,%r10d
    562e:	44 8b 64 24 1c       	mov    0x1c(%rsp),%r12d
    5633:	4c 8b 7c 24 30       	mov    0x30(%rsp),%r15
    5638:	44 8b 74 24 20       	mov    0x20(%rsp),%r14d
    563d:	44 89 54 24 1c       	mov    %r10d,0x1c(%rsp)
  if (edd > 0) {
    5642:	85 ed                	test   %ebp,%ebp
    5644:	0f 85 06 01 00 00    	jne    5750 <jerasure_bitmatrix_decode+0x310>
  for (i = 0; i < m; i++) {
    564a:	8b 4c 24 38          	mov    0x38(%rsp),%ecx
    564e:	49 63 c4             	movslq %r12d,%rax
    5651:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    5656:	85 c9                	test   %ecx,%ecx
    5658:	0f 8e ae 00 00 00    	jle    570c <jerasure_bitmatrix_decode+0x2cc>
    565e:	45 89 f5             	mov    %r14d,%r13d
    5661:	45 0f af ee          	imul   %r14d,%r13d
    5665:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    566a:	45 89 e0             	mov    %r12d,%r8d
    566d:	49 8d 1c 87          	lea    (%r15,%rax,4),%rbx
    5671:	45 0f af ec          	imul   %r12d,%r13d
    5675:	8b 44 24 38          	mov    0x38(%rsp),%eax
  decoding_matrix = NULL;
    5679:	31 ed                	xor    %ebp,%ebp
    567b:	44 01 e0             	add    %r12d,%eax
    567e:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
    5683:	44 89 74 24 1c       	mov    %r14d,0x1c(%rsp)
    5688:	41 89 c7             	mov    %eax,%r15d
    568b:	45 89 ee             	mov    %r13d,%r14d
    568e:	45 89 e5             	mov    %r12d,%r13d
    5691:	41 89 ec             	mov    %ebp,%r12d
    5694:	48 89 dd             	mov    %rbx,%rbp
    5697:	44 89 c3             	mov    %r8d,%ebx
    569a:	eb 12                	jmp    56ae <jerasure_bitmatrix_decode+0x26e>
    569c:	0f 1f 40 00          	nopl   0x0(%rax)
  for (i = 0; i < m; i++) {
    56a0:	ff c3                	inc    %ebx
    56a2:	48 83 c5 04          	add    $0x4,%rbp
    56a6:	45 01 f4             	add    %r14d,%r12d
    56a9:	41 39 df             	cmp    %ebx,%r15d
    56ac:	74 59                	je     5707 <jerasure_bitmatrix_decode+0x2c7>
    if (erased[k+i]) {
    56ae:	8b 55 00             	mov    0x0(%rbp),%edx
    56b1:	85 d2                	test   %edx,%edx
    56b3:	74 eb                	je     56a0 <jerasure_bitmatrix_decode+0x260>
      jerasure_bitmatrix_dotprod(k, w, bitmatrix+i*k*w*w, NULL, k+i, data_ptrs, coding_ptrs, size, packetsize);
    56b5:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    56ba:	49 63 d4             	movslq %r12d,%rdx
    56bd:	48 83 ec 08          	sub    $0x8,%rsp
    56c1:	48 8d 14 90          	lea    (%rax,%rdx,4),%rdx
    56c5:	8b 84 24 a0 00 00 00 	mov    0xa0(%rsp),%eax
    56cc:	41 89 d8             	mov    %ebx,%r8d
    56cf:	50                   	push   %rax
    56d0:	31 c9                	xor    %ecx,%ecx
    56d2:	44 89 ef             	mov    %r13d,%edi
    56d5:	8b 84 24 a0 00 00 00 	mov    0xa0(%rsp),%eax
    56dc:	ff c3                	inc    %ebx
    56de:	50                   	push   %rax
    56df:	48 83 c5 04          	add    $0x4,%rbp
    56e3:	45 01 f4             	add    %r14d,%r12d
    56e6:	ff b4 24 a0 00 00 00 	pushq  0xa0(%rsp)
    56ed:	4c 8b 8c 24 a0 00 00 	mov    0xa0(%rsp),%r9
    56f4:	00 
    56f5:	8b 74 24 3c          	mov    0x3c(%rsp),%esi
    56f9:	e8 c2 e0 ff ff       	callq  37c0 <jerasure_bitmatrix_dotprod>
    56fe:	48 83 c4 20          	add    $0x20,%rsp
  for (i = 0; i < m; i++) {
    5702:	41 39 df             	cmp    %ebx,%r15d
    5705:	75 a7                	jne    56ae <jerasure_bitmatrix_decode+0x26e>
    5707:	4c 8b 7c 24 20       	mov    0x20(%rsp),%r15
  free(erased);
    570c:	4c 89 ff             	mov    %r15,%rdi
    570f:	e8 6c bb ff ff       	callq  1280 <free@plt>
  if (dm_ids != NULL) free(dm_ids);
    5714:	48 83 3c 24 00       	cmpq   $0x0,(%rsp)
    5719:	74 09                	je     5724 <jerasure_bitmatrix_decode+0x2e4>
    571b:	48 8b 3c 24          	mov    (%rsp),%rdi
    571f:	e8 5c bb ff ff       	callq  1280 <free@plt>
  if (decoding_matrix != NULL) free(decoding_matrix);
    5724:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  return 0;
    5729:	45 31 e4             	xor    %r12d,%r12d
  if (decoding_matrix != NULL) free(decoding_matrix);
    572c:	48 85 ff             	test   %rdi,%rdi
    572f:	74 05                	je     5736 <jerasure_bitmatrix_decode+0x2f6>
    5731:	e8 4a bb ff ff       	callq  1280 <free@plt>
}
    5736:	48 83 c4 48          	add    $0x48,%rsp
    573a:	5b                   	pop    %rbx
    573b:	5d                   	pop    %rbp
    573c:	44 89 e0             	mov    %r12d,%eax
    573f:	41 5c                	pop    %r12
    5741:	41 5d                	pop    %r13
    5743:	41 5e                	pop    %r14
    5745:	41 5f                	pop    %r15
    5747:	c3                   	retq   
    5748:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    574f:	00 
    tmpids = talloc(int, k);
    5750:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    5755:	48 c1 e7 02          	shl    $0x2,%rdi
    5759:	e8 a2 bc ff ff       	callq  1400 <malloc@plt>
    for (i = 0; i < k; i++) {
    575e:	45 85 e4             	test   %r12d,%r12d
    5761:	44 8b 54 24 1c       	mov    0x1c(%rsp),%r10d
    tmpids = talloc(int, k);
    5766:	48 89 c5             	mov    %rax,%rbp
    for (i = 0; i < k; i++) {
    5769:	7e 26                	jle    5791 <jerasure_bitmatrix_decode+0x351>
    576b:	8b 4c 24 3c          	mov    0x3c(%rsp),%ecx
  decoding_matrix = NULL;
    576f:	31 d2                	xor    %edx,%edx
    5771:	eb 08                	jmp    577b <jerasure_bitmatrix_decode+0x33b>
    5773:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5778:	48 89 c2             	mov    %rax,%rdx
      tmpids[i] = (i < lastdrive) ? i : i+1;
    577b:	8d 42 01             	lea    0x1(%rdx),%eax
    577e:	41 39 d2             	cmp    %edx,%r10d
    5781:	0f 4f c2             	cmovg  %edx,%eax
    5784:	89 44 95 00          	mov    %eax,0x0(%rbp,%rdx,4)
    for (i = 0; i < k; i++) {
    5788:	48 8d 42 01          	lea    0x1(%rdx),%rax
    578c:	48 39 ca             	cmp    %rcx,%rdx
    578f:	75 e7                	jne    5778 <jerasure_bitmatrix_decode+0x338>
    jerasure_bitmatrix_dotprod(k, w, bitmatrix, tmpids, lastdrive, data_ptrs, coding_ptrs, size, packetsize);
    5791:	48 83 ec 08          	sub    $0x8,%rsp
    5795:	8b 84 24 a0 00 00 00 	mov    0xa0(%rsp),%eax
    579c:	44 89 e7             	mov    %r12d,%edi
    579f:	50                   	push   %rax
    57a0:	45 89 d0             	mov    %r10d,%r8d
    57a3:	48 89 e9             	mov    %rbp,%rcx
    57a6:	8b 84 24 a0 00 00 00 	mov    0xa0(%rsp),%eax
    57ad:	44 89 f6             	mov    %r14d,%esi
    57b0:	50                   	push   %rax
    57b1:	ff b4 24 a0 00 00 00 	pushq  0xa0(%rsp)
    57b8:	4c 8b 8c 24 a0 00 00 	mov    0xa0(%rsp),%r9
    57bf:	00 
    57c0:	48 8b 54 24 30       	mov    0x30(%rsp),%rdx
    57c5:	e8 f6 df ff ff       	callq  37c0 <jerasure_bitmatrix_dotprod>
    free(tmpids);
    57ca:	48 83 c4 20          	add    $0x20,%rsp
    57ce:	48 89 ef             	mov    %rbp,%rdi
    57d1:	e8 aa ba ff ff       	callq  1280 <free@plt>
    57d6:	e9 6f fe ff ff       	jmpq   564a <jerasure_bitmatrix_decode+0x20a>
    57db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    57e0:	49 63 c4             	movslq %r12d,%rax
    57e3:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    57e8:	48 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%rdi
    57ef:	00 
    57f0:	e9 f0 fc ff ff       	jmpq   54e5 <jerasure_bitmatrix_decode+0xa5>
    57f5:	0f 1f 00             	nopl   (%rax)
  if (row_k_ones != 1 || erased[k]) lastdrive = k;
    57f8:	49 63 c4             	movslq %r12d,%rax
    57fb:	45 8b 2c 87          	mov    (%r15,%rax,4),%r13d
    57ff:	45 85 ed             	test   %r13d,%r13d
    5802:	45 0f 45 d4          	cmovne %r12d,%r10d
    5806:	e9 b2 fc ff ff       	jmpq   54bd <jerasure_bitmatrix_decode+0x7d>
    580b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  dm_ids = NULL;
    5810:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
    5817:	00 
  decoding_matrix = NULL;
    5818:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    581f:	00 00 
    5821:	e9 24 fe ff ff       	jmpq   564a <jerasure_bitmatrix_decode+0x20a>
    5826:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    582d:	00 00 00 
  if (edd > 1 || (edd > 0 && (row_k_ones != 1 || erased[k]))) {
    5830:	45 8b 1c 87          	mov    (%r15,%rax,4),%r11d
    5834:	45 85 db             	test   %r11d,%r11d
    5837:	0f 85 a8 fc ff ff    	jne    54e5 <jerasure_bitmatrix_decode+0xa5>
  dm_ids = NULL;
    583d:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
    5844:	00 
  decoding_matrix = NULL;
    5845:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    584c:	00 00 
  for (i = 0; edd > 0 && i < lastdrive; i++) {
    584e:	45 85 d2             	test   %r10d,%r10d
    5851:	0f 85 1b fd ff ff    	jne    5572 <jerasure_bitmatrix_decode+0x132>
    5857:	44 89 54 24 1c       	mov    %r10d,0x1c(%rsp)
    tmpids = talloc(int, k);
    585c:	e8 9f bb ff ff       	callq  1400 <malloc@plt>
    5861:	48 89 c5             	mov    %rax,%rbp
    for (i = 0; i < k; i++) {
    5864:	44 8b 54 24 1c       	mov    0x1c(%rsp),%r10d
    5869:	e9 fd fe ff ff       	jmpq   576b <jerasure_bitmatrix_decode+0x32b>
    586e:	66 90                	xchg   %ax,%ax
  if (row_k_ones != 1 || erased[k]) lastdrive = k;
    5870:	83 fb 01             	cmp    $0x1,%ebx
    5873:	74 37                	je     58ac <jerasure_bitmatrix_decode+0x46c>
  for (i = 0; i < m; i++) {
    5875:	44 8b 44 24 38       	mov    0x38(%rsp),%r8d
    587a:	49 63 c4             	movslq %r12d,%rax
  dm_ids = NULL;
    587d:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
    5884:	00 
  decoding_matrix = NULL;
    5885:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    588c:	00 00 
    588e:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  for (i = 0; i < m; i++) {
    5893:	45 85 c0             	test   %r8d,%r8d
    5896:	0f 8f c2 fd ff ff    	jg     565e <jerasure_bitmatrix_decode+0x21e>
  free(erased);
    589c:	4c 89 ff             	mov    %r15,%rdi
    589f:	e8 dc b9 ff ff       	callq  1280 <free@plt>
  return 0;
    58a4:	45 31 e4             	xor    %r12d,%r12d
    58a7:	e9 8a fe ff ff       	jmpq   5736 <jerasure_bitmatrix_decode+0x2f6>
  if (row_k_ones != 1 || erased[k]) lastdrive = k;
    58ac:	49 63 c4             	movslq %r12d,%rax
    58af:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    58b4:	41 8b 04 87          	mov    (%r15,%rax,4),%eax
    58b8:	85 c0                	test   %eax,%eax
    58ba:	0f 85 50 ff ff ff    	jne    5810 <jerasure_bitmatrix_decode+0x3d0>
  for (i = 0; i < m; i++) {
    58c0:	8b 7c 24 38          	mov    0x38(%rsp),%edi
  dm_ids = NULL;
    58c4:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
    58cb:	00 
  decoding_matrix = NULL;
    58cc:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    58d3:	00 00 
  for (i = 0; i < m; i++) {
    58d5:	85 ff                	test   %edi,%edi
    58d7:	0f 8f 81 fd ff ff    	jg     565e <jerasure_bitmatrix_decode+0x21e>
    58dd:	eb bd                	jmp    589c <jerasure_bitmatrix_decode+0x45c>
  if (erased == NULL) return -1;
    58df:	41 bc ff ff ff ff    	mov    $0xffffffff,%r12d
    58e5:	e9 4c fe ff ff       	jmpq   5736 <jerasure_bitmatrix_decode+0x2f6>
      free(erased);
    58ea:	4c 89 ff             	mov    %r15,%rdi
    58ed:	e8 8e b9 ff ff       	callq  1280 <free@plt>
      free(dm_ids);
    58f2:	48 8b 3c 24          	mov    (%rsp),%rdi
      return -1;
    58f6:	41 bc ff ff ff ff    	mov    $0xffffffff,%r12d
      free(dm_ids);
    58fc:	e8 7f b9 ff ff       	callq  1280 <free@plt>
      free(decoding_matrix);
    5901:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    5906:	e8 75 b9 ff ff       	callq  1280 <free@plt>
      return -1;
    590b:	e9 26 fe ff ff       	jmpq   5736 <jerasure_bitmatrix_decode+0x2f6>
      free(erased);
    5910:	4c 89 ff             	mov    %r15,%rdi
    5913:	e8 68 b9 ff ff       	callq  1280 <free@plt>
      free(dm_ids);
    5918:	48 8b 3c 24          	mov    (%rsp),%rdi
      return -1;
    591c:	41 83 cc ff          	or     $0xffffffff,%r12d
      free(dm_ids);
    5920:	e8 5b b9 ff ff       	callq  1280 <free@plt>
      return -1;
    5925:	e9 0c fe ff ff       	jmpq   5736 <jerasure_bitmatrix_decode+0x2f6>
      free(erased);
    592a:	4c 89 ff             	mov    %r15,%rdi
    592d:	e8 4e b9 ff ff       	callq  1280 <free@plt>
      return -1;
    5932:	41 83 cc ff          	or     $0xffffffff,%r12d
    5936:	e9 fb fd ff ff       	jmpq   5736 <jerasure_bitmatrix_decode+0x2f6>
    593b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000005940 <jerasure_invertible_bitmatrix>:

int jerasure_invertible_bitmatrix(int *mat, int rows)
{
    5940:	f3 0f 1e fa          	endbr64 
    5944:	41 57                	push   %r15
    5946:	41 56                	push   %r14
    5948:	41 55                	push   %r13
    594a:	41 54                	push   %r12
    594c:	55                   	push   %rbp
    594d:	53                   	push   %rbx
    594e:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
 
  cols = rows;

  /* First -- convert into upper triangular */

  for (i = 0; i < cols; i++) {
    5953:	85 f6                	test   %esi,%esi
    5955:	0f 8e 64 01 00 00    	jle    5abf <jerasure_invertible_bitmatrix+0x17f>
    595b:	4c 8b 7c 24 e8       	mov    -0x18(%rsp),%r15
    5960:	44 8d 56 ff          	lea    -0x1(%rsi),%r10d
    5964:	48 63 ce             	movslq %esi,%rcx
    5967:	4d 8d 5f 04          	lea    0x4(%r15),%r11
    596b:	4d 89 fc             	mov    %r15,%r12
    596e:	4c 89 d2             	mov    %r10,%rdx
    5971:	4a 8d 3c 11          	lea    (%rcx,%r10,1),%rdi
    5975:	4c 89 54 24 f0       	mov    %r10,-0x10(%rsp)
    597a:	4c 89 5c 24 f8       	mov    %r11,-0x8(%rsp)
    597f:	49 f7 d2             	not    %r10

    /* Swap rows if we have a zero i,i element.  If we can't swap, then the 
       matrix was not invertible */

    if ((mat[i*cols+i]) == 0) { 
    5982:	45 8b 1c 24          	mov    (%r12),%r11d
    5986:	48 8d 04 8d 04 00 00 	lea    0x4(,%rcx,4),%rax
    598d:	00 
  for (i = 0; i < cols; i++) {
    598e:	31 db                	xor    %ebx,%ebx
    5990:	4d 89 d5             	mov    %r10,%r13
    5993:	48 89 44 24 e0       	mov    %rax,-0x20(%rsp)
    5998:	48 f7 da             	neg    %rdx
    599b:	48 8d 04 8d 00 00 00 	lea    0x0(,%rcx,4),%rax
    59a2:	00 
    59a3:	31 ed                	xor    %ebp,%ebp
    59a5:	45 31 f6             	xor    %r14d,%r14d
    59a8:	4d 8d 7c bf 04       	lea    0x4(%r15,%rdi,4),%r15
    59ad:	49 c1 e5 02          	shl    $0x2,%r13
    if ((mat[i*cols+i]) == 0) { 
    59b1:	ff c3                	inc    %ebx
    59b3:	45 85 db             	test   %r11d,%r11d
    59b6:	74 7b                	je     5a33 <jerasure_invertible_bitmatrix+0xf3>
    59b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    59bf:	00 
        tmp = mat[i*cols+k]; mat[i*cols+k] = mat[j*cols+k]; mat[j*cols+k] = tmp;
      }
    }
 
    /* Now for each j>i, add A_ji*Ai to Aj */
    for (j = i+1; j != rows; j++) {
    59c0:	39 de                	cmp    %ebx,%esi
    59c2:	0f 84 f7 00 00 00    	je     5abf <jerasure_invertible_bitmatrix+0x17f>
    59c8:	49 89 c9             	mov    %rcx,%r9
    59cb:	41 01 f6             	add    %esi,%r14d
    59ce:	4c 8d 14 29          	lea    (%rcx,%rbp,1),%r10
    59d2:	49 f7 d9             	neg    %r9
    59d5:	4d 8d 04 af          	lea    (%r15,%rbp,4),%r8
    59d9:	41 89 db             	mov    %ebx,%r11d
    59dc:	eb 10                	jmp    59ee <jerasure_invertible_bitmatrix+0xae>
    59de:	66 90                	xchg   %ax,%ax
    59e0:	41 ff c3             	inc    %r11d
    59e3:	49 29 c9             	sub    %rcx,%r9
    59e6:	49 01 c0             	add    %rax,%r8
    59e9:	44 39 de             	cmp    %r11d,%esi
    59ec:	74 2f                	je     5a1d <jerasure_invertible_bitmatrix+0xdd>
      if (mat[j*cols+i] != 0) {
    59ee:	41 8b 7c 90 fc       	mov    -0x4(%r8,%rdx,4),%edi
    59f3:	85 ff                	test   %edi,%edi
    59f5:	74 e9                	je     59e0 <jerasure_invertible_bitmatrix+0xa0>
    59f7:	4b 8d 7c 05 00       	lea    0x0(%r13,%r8,1),%rdi
    59fc:	0f 1f 40 00          	nopl   0x0(%rax)
        for (k = 0; k < cols; k++) {
          mat[j*cols+k] ^= mat[i*cols+k]; 
    5a00:	42 8b 2c 8f          	mov    (%rdi,%r9,4),%ebp
    5a04:	31 2f                	xor    %ebp,(%rdi)
        for (k = 0; k < cols; k++) {
    5a06:	48 83 c7 04          	add    $0x4,%rdi
    5a0a:	4c 39 c7             	cmp    %r8,%rdi
    5a0d:	75 f1                	jne    5a00 <jerasure_invertible_bitmatrix+0xc0>
    for (j = i+1; j != rows; j++) {
    5a0f:	41 ff c3             	inc    %r11d
    5a12:	49 29 c9             	sub    %rcx,%r9
    5a15:	49 01 c0             	add    %rax,%r8
    5a18:	44 39 de             	cmp    %r11d,%esi
    5a1b:	75 d1                	jne    59ee <jerasure_invertible_bitmatrix+0xae>
  for (i = 0; i < cols; i++) {
    5a1d:	4c 03 64 24 e0       	add    -0x20(%rsp),%r12
    if ((mat[i*cols+i]) == 0) { 
    5a22:	45 8b 1c 24          	mov    (%r12),%r11d
    5a26:	48 ff c2             	inc    %rdx
    5a29:	4c 89 d5             	mov    %r10,%rbp
    5a2c:	ff c3                	inc    %ebx
    5a2e:	45 85 db             	test   %r11d,%r11d
    5a31:	75 8d                	jne    59c0 <jerasure_invertible_bitmatrix+0x80>
      for (j = i+1; j < rows && (mat[j*cols+i]) == 0; j++) ;
    5a33:	39 de                	cmp    %ebx,%esi
    5a35:	0f 8e 98 00 00 00    	jle    5ad3 <jerasure_invertible_bitmatrix+0x193>
    5a3b:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
    5a40:	46 8d 04 36          	lea    (%rsi,%r14,1),%r8d
    5a44:	4c 8b 54 24 e8       	mov    -0x18(%rsp),%r10
    5a49:	4d 63 c8             	movslq %r8d,%r9
    5a4c:	48 01 d7             	add    %rdx,%rdi
    5a4f:	4c 01 cf             	add    %r9,%rdi
    5a52:	4d 8d 0c ba          	lea    (%r10,%rdi,4),%r9
    5a56:	41 89 da             	mov    %ebx,%r10d
    5a59:	eb 13                	jmp    5a6e <jerasure_invertible_bitmatrix+0x12e>
    5a5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5a60:	41 ff c2             	inc    %r10d
    5a63:	41 01 f0             	add    %esi,%r8d
    5a66:	49 01 c1             	add    %rax,%r9
    5a69:	44 39 d6             	cmp    %r10d,%esi
    5a6c:	74 57                	je     5ac5 <jerasure_invertible_bitmatrix+0x185>
    5a6e:	45 8b 19             	mov    (%r9),%r11d
    5a71:	44 89 c7             	mov    %r8d,%edi
    5a74:	45 85 db             	test   %r11d,%r11d
    5a77:	74 e7                	je     5a60 <jerasure_invertible_bitmatrix+0x120>
      for (k = 0; k < cols; k++) {
    5a79:	4c 8b 54 24 e8       	mov    -0x18(%rsp),%r10
    5a7e:	48 63 ff             	movslq %edi,%rdi
    5a81:	4d 8d 04 aa          	lea    (%r10,%rbp,4),%r8
    5a85:	4c 8b 54 24 f0       	mov    -0x10(%rsp),%r10
    5a8a:	48 29 ef             	sub    %rbp,%rdi
    5a8d:	4d 8d 0c 2a          	lea    (%r10,%rbp,1),%r9
    5a91:	4c 8b 54 24 f8       	mov    -0x8(%rsp),%r10
    5a96:	4f 8d 0c 8a          	lea    (%r10,%r9,4),%r9
    5a9a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
        tmp = mat[i*cols+k]; mat[i*cols+k] = mat[j*cols+k]; mat[j*cols+k] = tmp;
    5aa0:	45 8b 10             	mov    (%r8),%r10d
    5aa3:	45 8b 1c b8          	mov    (%r8,%rdi,4),%r11d
    5aa7:	45 89 18             	mov    %r11d,(%r8)
    5aaa:	45 89 14 b8          	mov    %r10d,(%r8,%rdi,4)
      for (k = 0; k < cols; k++) {
    5aae:	49 83 c0 04          	add    $0x4,%r8
    5ab2:	4d 39 c8             	cmp    %r9,%r8
    5ab5:	75 e9                	jne    5aa0 <jerasure_invertible_bitmatrix+0x160>
    for (j = i+1; j != rows; j++) {
    5ab7:	39 de                	cmp    %ebx,%esi
    5ab9:	0f 85 09 ff ff ff    	jne    59c8 <jerasure_invertible_bitmatrix+0x88>
        }
      }
    }
  }
  return 1;
    5abf:	41 bb 01 00 00 00    	mov    $0x1,%r11d
}
    5ac5:	5b                   	pop    %rbx
    5ac6:	5d                   	pop    %rbp
    5ac7:	41 5c                	pop    %r12
    5ac9:	41 5d                	pop    %r13
    5acb:	41 5e                	pop    %r14
    5acd:	44 89 d8             	mov    %r11d,%eax
    5ad0:	41 5f                	pop    %r15
    5ad2:	c3                   	retq   
      if (j == rows) return 0;
    5ad3:	74 f0                	je     5ac5 <jerasure_invertible_bitmatrix+0x185>
    5ad5:	89 f7                	mov    %esi,%edi
    5ad7:	0f af fb             	imul   %ebx,%edi
    5ada:	eb 9d                	jmp    5a79 <jerasure_invertible_bitmatrix+0x139>
    5adc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000005ae0 <jerasure_matrix_multiply>:

  
int *jerasure_matrix_multiply(int *m1, int *m2, int r1, int c1, int r2, int c2, int w)
{
    5ae0:	f3 0f 1e fa          	endbr64 
    5ae4:	41 57                	push   %r15
    5ae6:	41 56                	push   %r14
    5ae8:	41 55                	push   %r13
    5aea:	41 54                	push   %r12
    5aec:	55                   	push   %rbp
    5aed:	49 63 e9             	movslq %r9d,%rbp
    5af0:	49 89 ef             	mov    %rbp,%r15
    5af3:	53                   	push   %rbx
    5af4:	89 d3                	mov    %edx,%ebx
    5af6:	48 83 ec 58          	sub    $0x58,%rsp
    5afa:	48 89 7c 24 40       	mov    %rdi,0x40(%rsp)
    5aff:	89 6c 24 34          	mov    %ebp,0x34(%rsp)
  int *product, i, j, k, l;

  product = (int *) malloc(sizeof(int)*r1*c2);
    5b03:	48 63 fa             	movslq %edx,%rdi
    5b06:	48 c1 e5 02          	shl    $0x2,%rbp
    5b0a:	48 0f af fd          	imul   %rbp,%rdi
{
    5b0e:	89 54 24 3c          	mov    %edx,0x3c(%rsp)
    5b12:	48 89 74 24 28       	mov    %rsi,0x28(%rsp)
    5b17:	89 4c 24 48          	mov    %ecx,0x48(%rsp)
    5b1b:	44 89 44 24 10       	mov    %r8d,0x10(%rsp)
    5b20:	44 8b a4 24 90 00 00 	mov    0x90(%rsp),%r12d
    5b27:	00 
  product = (int *) malloc(sizeof(int)*r1*c2);
    5b28:	e8 d3 b8 ff ff       	callq  1400 <malloc@plt>
  for (i = 0; i < r1*c2; i++) product[i] = 0;
    5b2d:	89 da                	mov    %ebx,%edx
    5b2f:	41 0f af d7          	imul   %r15d,%edx
  product = (int *) malloc(sizeof(int)*r1*c2);
    5b33:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  for (i = 0; i < r1*c2; i++) product[i] = 0;
    5b38:	85 d2                	test   %edx,%edx
    5b3a:	7e 23                	jle    5b5f <jerasure_matrix_multiply+0x7f>
    5b3c:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
    5b41:	ff ca                	dec    %edx
    5b43:	48 89 f0             	mov    %rsi,%rax
    5b46:	48 8d 54 96 04       	lea    0x4(%rsi,%rdx,4),%rdx
    5b4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5b50:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
    5b56:	48 83 c0 04          	add    $0x4,%rax
    5b5a:	48 39 d0             	cmp    %rdx,%rax
    5b5d:	75 f1                	jne    5b50 <jerasure_matrix_multiply+0x70>

  for (i = 0; i < r1; i++) {
    5b5f:	8b 4c 24 3c          	mov    0x3c(%rsp),%ecx
    5b63:	85 c9                	test   %ecx,%ecx
    5b65:	0f 8e ef 00 00 00    	jle    5c5a <jerasure_matrix_multiply+0x17a>
    5b6b:	8b 44 24 34          	mov    0x34(%rsp),%eax
    5b6f:	c7 44 24 14 00 00 00 	movl   $0x0,0x14(%rsp)
    5b76:	00 
    5b77:	ff c8                	dec    %eax
    5b79:	c7 44 24 38 00 00 00 	movl   $0x0,0x38(%rsp)
    5b80:	00 
    5b81:	c7 44 24 30 00 00 00 	movl   $0x0,0x30(%rsp)
    5b88:	00 
    5b89:	89 44 24 4c          	mov    %eax,0x4c(%rsp)
    5b8d:	0f 1f 00             	nopl   (%rax)
    for (j = 0; j < c2; j++) {
    5b90:	8b 44 24 34          	mov    0x34(%rsp),%eax
    5b94:	85 c0                	test   %eax,%eax
    5b96:	0f 8e 9c 00 00 00    	jle    5c38 <jerasure_matrix_multiply+0x158>
    5b9c:	8b 44 24 4c          	mov    0x4c(%rsp),%eax
    5ba0:	48 63 54 24 38       	movslq 0x38(%rsp),%rdx
    5ba5:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    5baa:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    5baf:	48 8d 04 97          	lea    (%rdi,%rdx,4),%rax
    5bb3:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    5bb8:	8b 44 24 10          	mov    0x10(%rsp),%eax
    5bbc:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
    5bc3:	00 
    5bc4:	ff c8                	dec    %eax
    5bc6:	48 01 d0             	add    %rdx,%rax
    5bc9:	48 8d 5c 87 04       	lea    0x4(%rdi,%rax,4),%rbx
    5bce:	66 90                	xchg   %ax,%ax
      for (k = 0; k < r2; k++) {
    5bd0:	8b 54 24 10          	mov    0x10(%rsp),%edx
    5bd4:	85 d2                	test   %edx,%edx
    5bd6:	7e 46                	jle    5c1e <jerasure_matrix_multiply+0x13e>
        product[i*c2+j] ^= galois_single_multiply(m1[i*c1+k], m2[k*c2+j], w);
    5bd8:	48 8b 0c 24          	mov    (%rsp),%rcx
    5bdc:	8b 44 24 14          	mov    0x14(%rsp),%eax
    5be0:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    5be5:	01 c8                	add    %ecx,%eax
    5be7:	48 98                	cltq   
    5be9:	4c 8d 2c 87          	lea    (%rdi,%rax,4),%r13
    5bed:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    5bf2:	4c 8b 74 24 18       	mov    0x18(%rsp),%r14
    5bf7:	4c 8d 3c 88          	lea    (%rax,%rcx,4),%r15
    5bfb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5c00:	41 8b 37             	mov    (%r15),%esi
    5c03:	41 8b 3e             	mov    (%r14),%edi
    5c06:	44 89 e2             	mov    %r12d,%edx
    5c09:	e8 52 2a 00 00       	callq  8660 <galois_single_multiply>
    5c0e:	49 83 c6 04          	add    $0x4,%r14
    5c12:	41 31 45 00          	xor    %eax,0x0(%r13)
      for (k = 0; k < r2; k++) {
    5c16:	49 01 ef             	add    %rbp,%r15
    5c19:	49 39 de             	cmp    %rbx,%r14
    5c1c:	75 e2                	jne    5c00 <jerasure_matrix_multiply+0x120>
    for (j = 0; j < c2; j++) {
    5c1e:	48 8b 0c 24          	mov    (%rsp),%rcx
    5c22:	48 8d 41 01          	lea    0x1(%rcx),%rax
    5c26:	48 39 4c 24 08       	cmp    %rcx,0x8(%rsp)
    5c2b:	74 0b                	je     5c38 <jerasure_matrix_multiply+0x158>
    5c2d:	48 89 04 24          	mov    %rax,(%rsp)
    5c31:	eb 9d                	jmp    5bd0 <jerasure_matrix_multiply+0xf0>
    5c33:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  for (i = 0; i < r1; i++) {
    5c38:	ff 44 24 30          	incl   0x30(%rsp)
    5c3c:	8b 74 24 48          	mov    0x48(%rsp),%esi
    5c40:	01 74 24 38          	add    %esi,0x38(%rsp)
    5c44:	8b 44 24 30          	mov    0x30(%rsp),%eax
    5c48:	8b 74 24 34          	mov    0x34(%rsp),%esi
    5c4c:	01 74 24 14          	add    %esi,0x14(%rsp)
    5c50:	39 44 24 3c          	cmp    %eax,0x3c(%rsp)
    5c54:	0f 85 36 ff ff ff    	jne    5b90 <jerasure_matrix_multiply+0xb0>
      }
    }
  }
  return product;
}
    5c5a:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    5c5f:	48 83 c4 58          	add    $0x58,%rsp
    5c63:	5b                   	pop    %rbx
    5c64:	5d                   	pop    %rbp
    5c65:	41 5c                	pop    %r12
    5c67:	41 5d                	pop    %r13
    5c69:	41 5e                	pop    %r14
    5c6b:	41 5f                	pop    %r15
    5c6d:	c3                   	retq   
    5c6e:	66 90                	xchg   %ax,%ax

0000000000005c70 <jerasure_get_stats>:

void jerasure_get_stats(double *fill_in)
{
    5c70:	f3 0f 1e fa          	endbr64 
  fill_in[0] = jerasure_total_xor_bytes;
    5c74:	c5 fb 10 05 e4 b4 00 	vmovsd 0xb4e4(%rip),%xmm0        # 11160 <jerasure_total_xor_bytes>
    5c7b:	00 
  fill_in[1] = jerasure_total_gf_bytes;
  fill_in[2] = jerasure_total_memcpy_bytes;
  jerasure_total_xor_bytes = 0;
    5c7c:	48 c7 05 d9 b4 00 00 	movq   $0x0,0xb4d9(%rip)        # 11160 <jerasure_total_xor_bytes>
    5c83:	00 00 00 00 
  fill_in[0] = jerasure_total_xor_bytes;
    5c87:	c5 fb 11 07          	vmovsd %xmm0,(%rdi)
  fill_in[1] = jerasure_total_gf_bytes;
    5c8b:	c5 fb 10 05 c5 b4 00 	vmovsd 0xb4c5(%rip),%xmm0        # 11158 <jerasure_total_gf_bytes>
    5c92:	00 
  jerasure_total_gf_bytes = 0;
    5c93:	48 c7 05 ba b4 00 00 	movq   $0x0,0xb4ba(%rip)        # 11158 <jerasure_total_gf_bytes>
    5c9a:	00 00 00 00 
  fill_in[1] = jerasure_total_gf_bytes;
    5c9e:	c5 fb 11 47 08       	vmovsd %xmm0,0x8(%rdi)
  fill_in[2] = jerasure_total_memcpy_bytes;
    5ca3:	c5 fb 10 05 a5 b4 00 	vmovsd 0xb4a5(%rip),%xmm0        # 11150 <jerasure_total_memcpy_bytes>
    5caa:	00 
  jerasure_total_memcpy_bytes = 0;
    5cab:	48 c7 05 9a b4 00 00 	movq   $0x0,0xb49a(%rip)        # 11150 <jerasure_total_memcpy_bytes>
    5cb2:	00 00 00 00 
  fill_in[2] = jerasure_total_memcpy_bytes;
    5cb6:	c5 fb 11 47 10       	vmovsd %xmm0,0x10(%rdi)
}
    5cbb:	c3                   	retq   
    5cbc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000005cc0 <jerasure_do_scheduled_operations>:

void jerasure_do_scheduled_operations(char **ptrs, int **operations, int packetsize)
{
    5cc0:	f3 0f 1e fa          	endbr64 
  char *sptr;
  char *dptr;
  char * prev_dptr;
  int op, i;

  for (i = 0; i < packetsize; i += 512)
    5cc4:	85 d2                	test   %edx,%edx
    5cc6:	0f 8e e5 02 00 00    	jle    5fb1 <jerasure_do_scheduled_operations+0x2f1>
{
    5ccc:	41 55                	push   %r13
    5cce:	c5 fb 10 05 7a b4 00 	vmovsd 0xb47a(%rip),%xmm0        # 11150 <jerasure_total_memcpy_bytes>
    5cd5:	00 
    5cd6:	c5 fb 10 35 82 b4 00 	vmovsd 0xb482(%rip),%xmm6        # 11160 <jerasure_total_xor_bytes>
    5cdd:	00 
    5cde:	41 54                	push   %r12
    5ce0:	c5 fb 10 2d 98 4b 00 	vmovsd 0x4b98(%rip),%xmm5        # a880 <__PRETTY_FUNCTION__.5741+0x1a7>
    5ce7:	00 
  {
    prev_dptr = NULL;
    for (op = 0; operations[op][0] >= 0; op++) {
    5ce8:	45 31 d2             	xor    %r10d,%r10d
{
    5ceb:	55                   	push   %rbp
    for (op = 0; operations[op][0] >= 0; op++) {
    5cec:	45 31 db             	xor    %r11d,%r11d
{
    5cef:	53                   	push   %rbx
    5cf0:	48 89 f3             	mov    %rsi,%rbx
    for (op = 0; operations[op][0] >= 0; op++) {
    5cf3:	48 8b 2e             	mov    (%rsi),%rbp
    5cf6:	8b 75 00             	mov    0x0(%rbp),%esi
    5cf9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    prev_dptr = NULL;
    5d00:	31 c0                	xor    %eax,%eax
    for (op = 0; operations[op][0] >= 0; op++) {
    5d02:	85 f6                	test   %esi,%esi
    5d04:	0f 88 01 02 00 00    	js     5f0b <jerasure_do_scheduled_operations+0x24b>
    5d0a:	4c 8d 4b 08          	lea    0x8(%rbx),%r9
    5d0e:	4c 63 e6             	movslq %esi,%r12
    5d11:	49 89 e8             	mov    %rbp,%r8
    prev_dptr = NULL;
    5d14:	31 c0                	xor    %eax,%eax
    5d16:	e9 95 00 00 00       	jmpq   5db0 <jerasure_do_scheduled_operations+0xf0>
    5d1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
      sptr = ptrs[operations[op][0]] + operations[op][1]*packetsize + i;
      dptr = ptrs[operations[op][2]] + operations[op][3]*packetsize + i;
      if (operations[op][4]) {
        // do xor operation here
        __asm__ __volatile__ (
    5d20:	c5 fd ef 01          	vpxor  (%rcx),%ymm0,%ymm0
    5d24:	c5 f5 ef 49 20       	vpxor  0x20(%rcx),%ymm1,%ymm1
    5d29:	c5 ed ef 51 40       	vpxor  0x40(%rcx),%ymm2,%ymm2
    5d2e:	c5 e5 ef 59 60       	vpxor  0x60(%rcx),%ymm3,%ymm3
    5d33:	c5 dd ef a1 80 00 00 	vpxor  0x80(%rcx),%ymm4,%ymm4
    5d3a:	00 
    5d3b:	c5 d5 ef a9 a0 00 00 	vpxor  0xa0(%rcx),%ymm5,%ymm5
    5d42:	00 
    5d43:	c5 cd ef b1 c0 00 00 	vpxor  0xc0(%rcx),%ymm6,%ymm6
    5d4a:	00 
    5d4b:	c5 c5 ef b9 e0 00 00 	vpxor  0xe0(%rcx),%ymm7,%ymm7
    5d52:	00 
    5d53:	c5 3d ef 81 00 01 00 	vpxor  0x100(%rcx),%ymm8,%ymm8
    5d5a:	00 
    5d5b:	c5 35 ef 89 20 01 00 	vpxor  0x120(%rcx),%ymm9,%ymm9
    5d62:	00 
    5d63:	c5 2d ef 91 40 01 00 	vpxor  0x140(%rcx),%ymm10,%ymm10
    5d6a:	00 
    5d6b:	c5 25 ef 99 60 01 00 	vpxor  0x160(%rcx),%ymm11,%ymm11
    5d72:	00 
    5d73:	c5 1d ef a1 80 01 00 	vpxor  0x180(%rcx),%ymm12,%ymm12
    5d7a:	00 
    5d7b:	c5 15 ef a9 a0 01 00 	vpxor  0x1a0(%rcx),%ymm13,%ymm13
    5d82:	00 
    5d83:	c5 0d ef b1 c0 01 00 	vpxor  0x1c0(%rcx),%ymm14,%ymm14
    5d8a:	00 
    5d8b:	c5 05 ef b9 e0 01 00 	vpxor  0x1e0(%rcx),%ymm15,%ymm15
    5d92:	00 
    for (op = 0; operations[op][0] >= 0; op++) {
    5d93:	4d 8b 01             	mov    (%r9),%r8
          "vpxor  480(%0), %%ymm15, %%ymm15\n\t"
          :
          : "r"(sptr), "r"(dptr)
          : "ymm1", "ymm2", "ymm3", "ymm4"
        );
        jerasure_total_xor_bytes += 512;
    5d96:	c5 cb 58 f5          	vaddsd %xmm5,%xmm6,%xmm6
    for (op = 0; operations[op][0] >= 0; op++) {
    5d9a:	4d 63 20             	movslq (%r8),%r12
    5d9d:	41 bb 01 00 00 00    	mov    $0x1,%r11d
    5da3:	49 83 c1 08          	add    $0x8,%r9
    5da7:	45 85 e4             	test   %r12d,%r12d
    5daa:	0f 88 5b 01 00 00    	js     5f0b <jerasure_do_scheduled_operations+0x24b>
      sptr = ptrs[operations[op][0]] + operations[op][1]*packetsize + i;
    5db0:	41 8b 48 04          	mov    0x4(%r8),%ecx
      dptr = ptrs[operations[op][2]] + operations[op][3]*packetsize + i;
    5db4:	4d 63 68 08          	movslq 0x8(%r8),%r13
      sptr = ptrs[operations[op][0]] + operations[op][1]*packetsize + i;
    5db8:	0f af ca             	imul   %edx,%ecx
    5dbb:	48 63 c9             	movslq %ecx,%rcx
    5dbe:	4c 01 d1             	add    %r10,%rcx
    5dc1:	4a 03 0c e7          	add    (%rdi,%r12,8),%rcx
      dptr = ptrs[operations[op][2]] + operations[op][3]*packetsize + i;
    5dc5:	49 89 c4             	mov    %rax,%r12
    5dc8:	41 8b 40 0c          	mov    0xc(%r8),%eax
      if (operations[op][4]) {
    5dcc:	45 8b 40 10          	mov    0x10(%r8),%r8d
      dptr = ptrs[operations[op][2]] + operations[op][3]*packetsize + i;
    5dd0:	0f af c2             	imul   %edx,%eax
    5dd3:	48 98                	cltq   
    5dd5:	4c 01 d0             	add    %r10,%rax
    5dd8:	4a 03 04 ef          	add    (%rdi,%r13,8),%rax
      if (operations[op][4]) {
    5ddc:	45 85 c0             	test   %r8d,%r8d
    5ddf:	0f 85 3b ff ff ff    	jne    5d20 <jerasure_do_scheduled_operations+0x60>
      } else {
        // write previous result to memory and copy current data to last 8 ymm registers
        if (prev_dptr != NULL) {
    5de5:	4d 85 e4             	test   %r12,%r12
    5de8:	0f 84 93 00 00 00    	je     5e81 <jerasure_do_scheduled_operations+0x1c1>
          __asm__ __volatile__ (
    5dee:	c4 c1 7d 7f 04 24    	vmovdqa %ymm0,(%r12)
    5df4:	c4 c1 7d 7f 4c 24 20 	vmovdqa %ymm1,0x20(%r12)
    5dfb:	c4 c1 7d 7f 54 24 40 	vmovdqa %ymm2,0x40(%r12)
    5e02:	c4 c1 7d 7f 5c 24 60 	vmovdqa %ymm3,0x60(%r12)
    5e09:	c4 c1 7d 7f a4 24 80 	vmovdqa %ymm4,0x80(%r12)
    5e10:	00 00 00 
    5e13:	c4 c1 7d 7f ac 24 a0 	vmovdqa %ymm5,0xa0(%r12)
    5e1a:	00 00 00 
    5e1d:	c4 c1 7d 7f b4 24 c0 	vmovdqa %ymm6,0xc0(%r12)
    5e24:	00 00 00 
    5e27:	c4 c1 7d 7f bc 24 e0 	vmovdqa %ymm7,0xe0(%r12)
    5e2e:	00 00 00 
    5e31:	c4 41 7d 7f 84 24 00 	vmovdqa %ymm8,0x100(%r12)
    5e38:	01 00 00 
    5e3b:	c4 41 7d 7f 8c 24 20 	vmovdqa %ymm9,0x120(%r12)
    5e42:	01 00 00 
    5e45:	c4 41 7d 7f 94 24 40 	vmovdqa %ymm10,0x140(%r12)
    5e4c:	01 00 00 
    5e4f:	c4 41 7d 7f 9c 24 60 	vmovdqa %ymm11,0x160(%r12)
    5e56:	01 00 00 
    5e59:	c4 41 7d 7f a4 24 80 	vmovdqa %ymm12,0x180(%r12)
    5e60:	01 00 00 
    5e63:	c4 41 7d 7f ac 24 a0 	vmovdqa %ymm13,0x1a0(%r12)
    5e6a:	01 00 00 
    5e6d:	c4 41 7d 7f b4 24 c0 	vmovdqa %ymm14,0x1c0(%r12)
    5e74:	01 00 00 
    5e77:	c4 41 7d 7f bc 24 e0 	vmovdqa %ymm15,0x1e0(%r12)
    5e7e:	01 00 00 
            :
            : "r"(sptr), "r"(prev_dptr)
            : "ymm1", "ymm2", "ymm3", "ymm4"
          );
        }
        __asm__ __volatile__ (
    5e81:	c5 fd 6f 01          	vmovdqa (%rcx),%ymm0
    5e85:	c5 fd 6f 49 20       	vmovdqa 0x20(%rcx),%ymm1
    5e8a:	c5 fd 6f 51 40       	vmovdqa 0x40(%rcx),%ymm2
    5e8f:	c5 fd 6f 59 60       	vmovdqa 0x60(%rcx),%ymm3
    5e94:	c5 fd 6f a1 80 00 00 	vmovdqa 0x80(%rcx),%ymm4
    5e9b:	00 
    5e9c:	c5 fd 6f a9 a0 00 00 	vmovdqa 0xa0(%rcx),%ymm5
    5ea3:	00 
    5ea4:	c5 fd 6f b1 c0 00 00 	vmovdqa 0xc0(%rcx),%ymm6
    5eab:	00 
    5eac:	c5 fd 6f b9 e0 00 00 	vmovdqa 0xe0(%rcx),%ymm7
    5eb3:	00 
    5eb4:	c5 7d 6f 81 00 01 00 	vmovdqa 0x100(%rcx),%ymm8
    5ebb:	00 
    5ebc:	c5 7d 6f 89 20 01 00 	vmovdqa 0x120(%rcx),%ymm9
    5ec3:	00 
    5ec4:	c5 7d 6f 91 40 01 00 	vmovdqa 0x140(%rcx),%ymm10
    5ecb:	00 
    5ecc:	c5 7d 6f 99 60 01 00 	vmovdqa 0x160(%rcx),%ymm11
    5ed3:	00 
    5ed4:	c5 7d 6f a1 80 01 00 	vmovdqa 0x180(%rcx),%ymm12
    5edb:	00 
    5edc:	c5 7d 6f a9 a0 01 00 	vmovdqa 0x1a0(%rcx),%ymm13
    5ee3:	00 
    5ee4:	c5 7d 6f b1 c0 01 00 	vmovdqa 0x1c0(%rcx),%ymm14
    5eeb:	00 
    5eec:	c5 7d 6f b9 e0 01 00 	vmovdqa 0x1e0(%rcx),%ymm15
    5ef3:	00 
    for (op = 0; operations[op][0] >= 0; op++) {
    5ef4:	4d 8b 01             	mov    (%r9),%r8
          "vmovdqa  480(%0), %%ymm15\n\t"
          :
          : "r"(sptr), "r"(dptr)
          : "ymm1", "ymm2", "ymm3", "ymm4"
        );
        jerasure_total_memcpy_bytes += 512;
    5ef7:	c5 fb 58 c5          	vaddsd %xmm5,%xmm0,%xmm0
    for (op = 0; operations[op][0] >= 0; op++) {
    5efb:	4d 63 20             	movslq (%r8),%r12
    5efe:	49 83 c1 08          	add    $0x8,%r9
    5f02:	45 85 e4             	test   %r12d,%r12d
    5f05:	0f 89 a5 fe ff ff    	jns    5db0 <jerasure_do_scheduled_operations+0xf0>
      }
      prev_dptr = dptr;
    }
    // don't forget write last result
    __asm__ __volatile__ (
    5f0b:	c5 fd 7f 00          	vmovdqa %ymm0,(%rax)
    5f0f:	c5 fd 7f 48 20       	vmovdqa %ymm1,0x20(%rax)
    5f14:	c5 fd 7f 50 40       	vmovdqa %ymm2,0x40(%rax)
    5f19:	c5 fd 7f 58 60       	vmovdqa %ymm3,0x60(%rax)
    5f1e:	c5 fd 7f a0 80 00 00 	vmovdqa %ymm4,0x80(%rax)
    5f25:	00 
    5f26:	c5 fd 7f a8 a0 00 00 	vmovdqa %ymm5,0xa0(%rax)
    5f2d:	00 
    5f2e:	c5 fd 7f b0 c0 00 00 	vmovdqa %ymm6,0xc0(%rax)
    5f35:	00 
    5f36:	c5 fd 7f b8 e0 00 00 	vmovdqa %ymm7,0xe0(%rax)
    5f3d:	00 
    5f3e:	c5 7d 7f 80 00 01 00 	vmovdqa %ymm8,0x100(%rax)
    5f45:	00 
    5f46:	c5 7d 7f 88 20 01 00 	vmovdqa %ymm9,0x120(%rax)
    5f4d:	00 
    5f4e:	c5 7d 7f 90 40 01 00 	vmovdqa %ymm10,0x140(%rax)
    5f55:	00 
    5f56:	c5 7d 7f 98 60 01 00 	vmovdqa %ymm11,0x160(%rax)
    5f5d:	00 
    5f5e:	c5 7d 7f a0 80 01 00 	vmovdqa %ymm12,0x180(%rax)
    5f65:	00 
    5f66:	c5 7d 7f a8 a0 01 00 	vmovdqa %ymm13,0x1a0(%rax)
    5f6d:	00 
    5f6e:	c5 7d 7f b0 c0 01 00 	vmovdqa %ymm14,0x1c0(%rax)
    5f75:	00 
    5f76:	c5 7d 7f b8 e0 01 00 	vmovdqa %ymm15,0x1e0(%rax)
    5f7d:	00 
      "vmovdqa  %%ymm15, 480(%1)\n\t"
      :
      : "r"(sptr), "r"(prev_dptr)
      : "ymm1", "ymm2", "ymm3", "ymm4"
    );
    jerasure_total_memcpy_bytes += 512;
    5f7e:	49 81 c2 00 02 00 00 	add    $0x200,%r10
    5f85:	c5 fb 58 c5          	vaddsd %xmm5,%xmm0,%xmm0
  for (i = 0; i < packetsize; i += 512)
    5f89:	44 39 d2             	cmp    %r10d,%edx
    5f8c:	0f 8f 6e fd ff ff    	jg     5d00 <jerasure_do_scheduled_operations+0x40>
    5f92:	45 84 db             	test   %r11b,%r11b
    5f95:	74 08                	je     5f9f <jerasure_do_scheduled_operations+0x2df>
    5f97:	c5 fb 11 35 c1 b1 00 	vmovsd %xmm6,0xb1c1(%rip)        # 11160 <jerasure_total_xor_bytes>
    5f9e:	00 
    5f9f:	c5 fb 11 05 a9 b1 00 	vmovsd %xmm0,0xb1a9(%rip)        # 11150 <jerasure_total_memcpy_bytes>
    5fa6:	00 
    5fa7:	c5 f8 77             	vzeroupper 
  }
}
    5faa:	5b                   	pop    %rbx
    5fab:	5d                   	pop    %rbp
    5fac:	41 5c                	pop    %r12
    5fae:	41 5d                	pop    %r13
    5fb0:	c3                   	retq   
    5fb1:	c3                   	retq   
    5fb2:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    5fb9:	00 00 00 00 
    5fbd:	0f 1f 00             	nopl   (%rax)

0000000000005fc0 <jerasure_schedule_decode_cache>:
{
    5fc0:	f3 0f 1e fa          	endbr64 
    5fc4:	41 57                	push   %r15
    5fc6:	49 89 ca             	mov    %rcx,%r10
    5fc9:	4c 89 c9             	mov    %r9,%rcx
    5fcc:	41 56                	push   %r14
    5fce:	41 55                	push   %r13
    5fd0:	41 54                	push   %r12
    5fd2:	55                   	push   %rbp
    5fd3:	53                   	push   %rbx
    5fd4:	89 d3                	mov    %edx,%ebx
    5fd6:	4c 89 c2             	mov    %r8,%rdx
    5fd9:	48 83 ec 18          	sub    $0x18,%rsp
  if (erasures[1] == -1) {
    5fdd:	45 8b 40 04          	mov    0x4(%r8),%r8d
{
    5fe1:	44 8b 6c 24 60       	mov    0x60(%rsp),%r13d
  if (erasures[1] == -1) {
    5fe6:	41 83 f8 ff          	cmp    $0xffffffff,%r8d
    5fea:	0f 84 96 00 00 00    	je     6086 <jerasure_schedule_decode_cache+0xc6>
  } else if (erasures[2] == -1) {
    5ff0:	83 7a 08 ff          	cmpl   $0xffffffff,0x8(%rdx)
    5ff4:	0f 85 a0 00 00 00    	jne    609a <jerasure_schedule_decode_cache+0xda>
    index = erasures[0]*(k+m) + erasures[1];
    5ffa:	8b 02                	mov    (%rdx),%eax
    5ffc:	8d 2c 37             	lea    (%rdi,%rsi,1),%ebp
    5fff:	0f af c5             	imul   %ebp,%eax
    6002:	44 01 c0             	add    %r8d,%eax
  ptrs = set_up_ptrs_for_scheduled_decoding(k, m, erasures, data_ptrs, coding_ptrs);
    6005:	4c 8b 44 24 50       	mov    0x50(%rsp),%r8
  schedule = scache[index];
    600a:	48 98                	cltq   
    600c:	4d 8b 34 c2          	mov    (%r10,%rax,8),%r14
  ptrs = set_up_ptrs_for_scheduled_decoding(k, m, erasures, data_ptrs, coding_ptrs);
    6010:	e8 4b e5 ff ff       	callq  4560 <set_up_ptrs_for_scheduled_decoding>
    6015:	48 89 c7             	mov    %rax,%rdi
  if (ptrs == NULL) return -1;
    6018:	48 85 c0             	test   %rax,%rax
    601b:	74 7d                	je     609a <jerasure_schedule_decode_cache+0xda>
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    601d:	8b 44 24 58          	mov    0x58(%rsp),%eax
    6021:	85 c0                	test   %eax,%eax
    6023:	7e 4b                	jle    6070 <jerasure_schedule_decode_cache+0xb0>
    for (i = 0; i < k+m; i++) ptrs[i] += (packetsize*w);
    6025:	41 0f af dd          	imul   %r13d,%ebx
    6029:	8d 45 ff             	lea    -0x1(%rbp),%eax
    602c:	4c 8d 7c c7 08       	lea    0x8(%rdi,%rax,8),%r15
    6031:	89 5c 24 0c          	mov    %ebx,0xc(%rsp)
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    6035:	45 31 e4             	xor    %r12d,%r12d
    for (i = 0; i < k+m; i++) ptrs[i] += (packetsize*w);
    6038:	48 63 db             	movslq %ebx,%rbx
    603b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  jerasure_do_scheduled_operations(ptrs, schedule, packetsize);
    6040:	44 89 ea             	mov    %r13d,%edx
    6043:	4c 89 f6             	mov    %r14,%rsi
    6046:	e8 75 fc ff ff       	callq  5cc0 <jerasure_do_scheduled_operations>
    for (i = 0; i < k+m; i++) ptrs[i] += (packetsize*w);
    604b:	48 89 fa             	mov    %rdi,%rdx
    604e:	85 ed                	test   %ebp,%ebp
    6050:	7e 12                	jle    6064 <jerasure_schedule_decode_cache+0xa4>
    6052:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    6058:	48 01 1a             	add    %rbx,(%rdx)
    605b:	48 83 c2 08          	add    $0x8,%rdx
    605f:	49 39 d7             	cmp    %rdx,%r15
    6062:	75 f4                	jne    6058 <jerasure_schedule_decode_cache+0x98>
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    6064:	44 03 64 24 0c       	add    0xc(%rsp),%r12d
    6069:	44 39 64 24 58       	cmp    %r12d,0x58(%rsp)
    606e:	7f d0                	jg     6040 <jerasure_schedule_decode_cache+0x80>
  free(ptrs);
    6070:	e8 0b b2 ff ff       	callq  1280 <free@plt>
  return 0;
    6075:	31 c0                	xor    %eax,%eax
}
    6077:	48 83 c4 18          	add    $0x18,%rsp
    607b:	5b                   	pop    %rbx
    607c:	5d                   	pop    %rbp
    607d:	41 5c                	pop    %r12
    607f:	41 5d                	pop    %r13
    6081:	41 5e                	pop    %r14
    6083:	41 5f                	pop    %r15
    6085:	c3                   	retq   
    index = erasures[0]*(k+m) + erasures[0];
    6086:	44 8b 02             	mov    (%rdx),%r8d
    6089:	8d 2c 37             	lea    (%rdi,%rsi,1),%ebp
    608c:	89 e8                	mov    %ebp,%eax
    608e:	41 0f af c0          	imul   %r8d,%eax
    6092:	44 01 c0             	add    %r8d,%eax
    6095:	e9 6b ff ff ff       	jmpq   6005 <jerasure_schedule_decode_cache+0x45>
    return -1;
    609a:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    609f:	eb d6                	jmp    6077 <jerasure_schedule_decode_cache+0xb7>
    60a1:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    60a8:	00 00 00 00 
    60ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000060b0 <jerasure_schedule_encode>:

void jerasure_schedule_encode(int k, int m, int w, int **schedule,
                                   char **data_ptrs, char **coding_ptrs, int size, int packetsize)
{
    60b0:	f3 0f 1e fa          	endbr64 
    60b4:	41 57                	push   %r15
    60b6:	4c 63 ff             	movslq %edi,%r15
    60b9:	41 56                	push   %r14
    60bb:	41 55                	push   %r13
    60bd:	41 89 f5             	mov    %esi,%r13d
    60c0:	41 54                	push   %r12
  char **ptr_copy;
  int i, j, tdone;

  ptr_copy = talloc(char *, (k+m));
    60c2:	45 8d 24 37          	lea    (%r15,%rsi,1),%r12d
    60c6:	49 63 fc             	movslq %r12d,%rdi
{
    60c9:	55                   	push   %rbp
  ptr_copy = talloc(char *, (k+m));
    60ca:	48 c1 e7 03          	shl    $0x3,%rdi
{
    60ce:	48 89 cd             	mov    %rcx,%rbp
    60d1:	53                   	push   %rbx
    60d2:	89 d3                	mov    %edx,%ebx
    60d4:	48 83 ec 28          	sub    $0x28,%rsp
    60d8:	8b 44 24 60          	mov    0x60(%rsp),%eax
    60dc:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
    60e1:	4c 89 4c 24 10       	mov    %r9,0x10(%rsp)
    60e6:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    60ea:	44 8b 74 24 68       	mov    0x68(%rsp),%r14d
  ptr_copy = talloc(char *, (k+m));
    60ef:	e8 0c b3 ff ff       	callq  1400 <malloc@plt>
    60f4:	31 d2                	xor    %edx,%edx
  for (i = 0; i < k; i++) ptr_copy[i] = data_ptrs[i];
    60f6:	45 85 ff             	test   %r15d,%r15d
    60f9:	4c 8b 4c 24 10       	mov    0x10(%rsp),%r9
    60fe:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  ptr_copy = talloc(char *, (k+m));
    6103:	48 89 c7             	mov    %rax,%rdi
  for (i = 0; i < k; i++) ptr_copy[i] = data_ptrs[i];
    6106:	41 8d 4f ff          	lea    -0x1(%r15),%ecx
    610a:	7e 17                	jle    6123 <jerasure_schedule_encode+0x73>
    610c:	0f 1f 40 00          	nopl   0x0(%rax)
    6110:	49 8b 04 d0          	mov    (%r8,%rdx,8),%rax
    6114:	48 89 04 d7          	mov    %rax,(%rdi,%rdx,8)
    6118:	48 89 d0             	mov    %rdx,%rax
    611b:	48 ff c2             	inc    %rdx
    611e:	48 39 c1             	cmp    %rax,%rcx
    6121:	75 ed                	jne    6110 <jerasure_schedule_encode+0x60>
  for (i = 0; i < m; i++) ptr_copy[i+k] = coding_ptrs[i];
    6123:	45 85 ed             	test   %r13d,%r13d
    6126:	7e 23                	jle    614b <jerasure_schedule_encode+0x9b>
    6128:	41 8d 75 ff          	lea    -0x1(%r13),%esi
    612c:	4a 8d 0c ff          	lea    (%rdi,%r15,8),%rcx
    6130:	31 d2                	xor    %edx,%edx
    6132:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    6138:	49 8b 04 d1          	mov    (%r9,%rdx,8),%rax
    613c:	48 89 04 d1          	mov    %rax,(%rcx,%rdx,8)
    6140:	48 89 d0             	mov    %rdx,%rax
    6143:	48 ff c2             	inc    %rdx
    6146:	48 39 c6             	cmp    %rax,%rsi
    6149:	75 ed                	jne    6138 <jerasure_schedule_encode+0x88>
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    614b:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    614f:	85 c0                	test   %eax,%eax
    6151:	7e 4d                	jle    61a0 <jerasure_schedule_encode+0xf0>
    jerasure_do_scheduled_operations(ptr_copy, schedule, packetsize);
    for (i = 0; i < k+m; i++) ptr_copy[i] += (packetsize*w);
    6153:	41 0f af de          	imul   %r14d,%ebx
    6157:	41 8d 44 24 ff       	lea    -0x1(%r12),%eax
    615c:	4c 8d 7c c7 08       	lea    0x8(%rdi,%rax,8),%r15
    6161:	89 5c 24 10          	mov    %ebx,0x10(%rsp)
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    6165:	45 31 ed             	xor    %r13d,%r13d
    for (i = 0; i < k+m; i++) ptr_copy[i] += (packetsize*w);
    6168:	48 63 db             	movslq %ebx,%rbx
    616b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    jerasure_do_scheduled_operations(ptr_copy, schedule, packetsize);
    6170:	44 89 f2             	mov    %r14d,%edx
    6173:	48 89 ee             	mov    %rbp,%rsi
    6176:	e8 45 fb ff ff       	callq  5cc0 <jerasure_do_scheduled_operations>
    for (i = 0; i < k+m; i++) ptr_copy[i] += (packetsize*w);
    617b:	48 89 fa             	mov    %rdi,%rdx
    617e:	45 85 e4             	test   %r12d,%r12d
    6181:	7e 11                	jle    6194 <jerasure_schedule_encode+0xe4>
    6183:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    6188:	48 01 1a             	add    %rbx,(%rdx)
    618b:	48 83 c2 08          	add    $0x8,%rdx
    618f:	49 39 d7             	cmp    %rdx,%r15
    6192:	75 f4                	jne    6188 <jerasure_schedule_encode+0xd8>
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    6194:	44 03 6c 24 10       	add    0x10(%rsp),%r13d
    6199:	44 39 6c 24 0c       	cmp    %r13d,0xc(%rsp)
    619e:	7f d0                	jg     6170 <jerasure_schedule_encode+0xc0>
  }
  free(ptr_copy);
}
    61a0:	48 83 c4 28          	add    $0x28,%rsp
    61a4:	5b                   	pop    %rbx
    61a5:	5d                   	pop    %rbp
    61a6:	41 5c                	pop    %r12
    61a8:	41 5d                	pop    %r13
    61aa:	41 5e                	pop    %r14
    61ac:	41 5f                	pop    %r15
  free(ptr_copy);
    61ae:	e9 cd b0 ff ff       	jmpq   1280 <free@plt>
    61b3:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    61ba:	00 00 00 00 
    61be:	66 90                	xchg   %ax,%ax

00000000000061c0 <jerasure_dumb_bitmatrix_to_schedule>:
    
int **jerasure_dumb_bitmatrix_to_schedule(int k, int m, int w, int *bitmatrix)
{
    61c0:	f3 0f 1e fa          	endbr64 
    61c4:	41 57                	push   %r15
    61c6:	41 56                	push   %r14
    61c8:	41 89 fe             	mov    %edi,%r14d
    61cb:	41 55                	push   %r13
    61cd:	41 54                	push   %r12
    61cf:	55                   	push   %rbp
    61d0:	89 d5                	mov    %edx,%ebp
    61d2:	53                   	push   %rbx
    61d3:	89 f3                	mov    %esi,%ebx

  operations = talloc(int *, k*m*w*w+1);
  op = 0;
  
  index = 0;
  for (i = 0; i < m*w; i++) {
    61d5:	0f af dd             	imul   %ebp,%ebx
{
    61d8:	48 83 ec 48          	sub    $0x48,%rsp
    61dc:	89 7c 24 20          	mov    %edi,0x20(%rsp)
  operations = talloc(int *, k*m*w*w+1);
    61e0:	0f af fe             	imul   %esi,%edi
{
    61e3:	89 54 24 24          	mov    %edx,0x24(%rsp)
    61e7:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
  operations = talloc(int *, k*m*w*w+1);
    61ec:	0f af fa             	imul   %edx,%edi
    61ef:	0f af fa             	imul   %edx,%edi
    61f2:	ff c7                	inc    %edi
    61f4:	48 63 ff             	movslq %edi,%rdi
    61f7:	48 c1 e7 03          	shl    $0x3,%rdi
    61fb:	e8 00 b2 ff ff       	callq  1400 <malloc@plt>
    6200:	49 89 c7             	mov    %rax,%r15
  for (i = 0; i < m*w; i++) {
    6203:	89 5c 24 38          	mov    %ebx,0x38(%rsp)
    6207:	85 db                	test   %ebx,%ebx
    6209:	0f 8e 3a 01 00 00    	jle    6349 <jerasure_dumb_bitmatrix_to_schedule+0x189>
    optodo = 0;
    for (j = 0; j < k*w; j++) {
    620f:	44 0f af f5          	imul   %ebp,%r14d
  for (i = 0; i < m*w; i++) {
    6213:	c7 44 24 10 00 00 00 	movl   $0x0,0x10(%rsp)
    621a:	00 
  index = 0;
    621b:	c7 44 24 2c 00 00 00 	movl   $0x0,0x2c(%rsp)
    6222:	00 
    6223:	41 8d 46 ff          	lea    -0x1(%r14),%eax
    6227:	89 44 24 3c          	mov    %eax,0x3c(%rsp)
  op = 0;
    622b:	45 31 ed             	xor    %r13d,%r13d
    for (j = 0; j < k*w; j++) {
    622e:	44 89 74 24 28       	mov    %r14d,0x28(%rsp)
    6233:	4d 89 fe             	mov    %r15,%r14
    6236:	45 89 ef             	mov    %r13d,%r15d
    6239:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    6240:	8b 44 24 28          	mov    0x28(%rsp),%eax
    6244:	85 c0                	test   %eax,%eax
    6246:	0f 8e f4 00 00 00    	jle    6340 <jerasure_dumb_bitmatrix_to_schedule+0x180>
    624c:	44 8b 54 24 3c       	mov    0x3c(%rsp),%r10d
    6251:	48 63 44 24 2c       	movslq 0x2c(%rsp),%rax
    6256:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    optodo = 0;
    625b:	45 31 e4             	xor    %r12d,%r12d
    625e:	4c 8d 0c 81          	lea    (%rcx,%rax,4),%r9
    for (j = 0; j < k*w; j++) {
    6262:	31 db                	xor    %ebx,%ebx
    6264:	4d 89 d5             	mov    %r10,%r13
    6267:	44 89 e2             	mov    %r12d,%edx
    626a:	eb 10                	jmp    627c <jerasure_dumb_bitmatrix_to_schedule+0xbc>
    626c:	0f 1f 40 00          	nopl   0x0(%rax)
    6270:	48 8d 43 01          	lea    0x1(%rbx),%rax
    6274:	49 39 dd             	cmp    %rbx,%r13
    6277:	74 7a                	je     62f3 <jerasure_dumb_bitmatrix_to_schedule+0x133>
    6279:	48 89 c3             	mov    %rax,%rbx
      if (bitmatrix[index]) {
    627c:	49 63 ef             	movslq %r15d,%rbp
    627f:	41 8b 0c 99          	mov    (%r9,%rbx,4),%ecx
    6283:	48 c1 e5 03          	shl    $0x3,%rbp
    6287:	49 8d 34 2e          	lea    (%r14,%rbp,1),%rsi
    628b:	85 c9                	test   %ecx,%ecx
    628d:	74 e1                	je     6270 <jerasure_dumb_bitmatrix_to_schedule+0xb0>
        operations[op] = talloc(int, 5);
    628f:	bf 14 00 00 00       	mov    $0x14,%edi
    6294:	48 89 74 24 18       	mov    %rsi,0x18(%rsp)
    6299:	89 54 24 14          	mov    %edx,0x14(%rsp)
    629d:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
    62a2:	e8 59 b1 ff ff       	callq  1400 <malloc@plt>
    62a7:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
        operations[op][4] = optodo;
    62ac:	8b 54 24 14          	mov    0x14(%rsp),%edx
        operations[op] = talloc(int, 5);
    62b0:	48 89 c1             	mov    %rax,%rcx
        operations[op][4] = optodo;
    62b3:	89 50 10             	mov    %edx,0x10(%rax)
        operations[op] = talloc(int, 5);
    62b6:	48 89 06             	mov    %rax,(%rsi)
        operations[op][0] = j/w;
    62b9:	8b 7c 24 24          	mov    0x24(%rsp),%edi
    62bd:	89 d8                	mov    %ebx,%eax
    62bf:	99                   	cltd   
    62c0:	f7 ff                	idiv   %edi
        operations[op][1] = j%w;
        operations[op][2] = k+i/w;
        operations[op][3] = i%w;
        optodo = 1;
    62c2:	4c 8b 4c 24 08       	mov    0x8(%rsp),%r9
        op++;
    62c7:	41 ff c7             	inc    %r15d
    62ca:	49 8d 74 2e 08       	lea    0x8(%r14,%rbp,1),%rsi
        operations[op][0] = j/w;
    62cf:	89 01                	mov    %eax,(%rcx)
        operations[op][1] = j%w;
    62d1:	8b 44 24 10          	mov    0x10(%rsp),%eax
    62d5:	89 51 04             	mov    %edx,0x4(%rcx)
        operations[op][2] = k+i/w;
    62d8:	99                   	cltd   
    62d9:	f7 ff                	idiv   %edi
    62db:	03 44 24 20          	add    0x20(%rsp),%eax
    62df:	89 41 08             	mov    %eax,0x8(%rcx)
        operations[op][3] = i%w;
    62e2:	89 51 0c             	mov    %edx,0xc(%rcx)
        op++;
    62e5:	48 8d 43 01          	lea    0x1(%rbx),%rax
        optodo = 1;
    62e9:	ba 01 00 00 00       	mov    $0x1,%edx
    for (j = 0; j < k*w; j++) {
    62ee:	49 39 dd             	cmp    %rbx,%r13
    62f1:	75 86                	jne    6279 <jerasure_dumb_bitmatrix_to_schedule+0xb9>
    62f3:	8b 54 24 28          	mov    0x28(%rsp),%edx
    62f7:	49 89 f5             	mov    %rsi,%r13
    62fa:	01 54 24 2c          	add    %edx,0x2c(%rsp)
  for (i = 0; i < m*w; i++) {
    62fe:	ff 44 24 10          	incl   0x10(%rsp)
    6302:	8b 44 24 10          	mov    0x10(%rsp),%eax
    6306:	3b 44 24 38          	cmp    0x38(%rsp),%eax
    630a:	0f 85 30 ff ff ff    	jne    6240 <jerasure_dumb_bitmatrix_to_schedule+0x80>
    6310:	4d 89 f7             	mov    %r14,%r15
    6313:	4d 89 ee             	mov    %r13,%r14
        
      }
      index++;
    }
  }
  operations[op] = talloc(int, 5);
    6316:	bf 14 00 00 00       	mov    $0x14,%edi
    631b:	e8 e0 b0 ff ff       	callq  1400 <malloc@plt>
    6320:	49 89 06             	mov    %rax,(%r14)
  operations[op][0] = -1;
    6323:	c7 00 ff ff ff ff    	movl   $0xffffffff,(%rax)
  return operations;
}
    6329:	48 83 c4 48          	add    $0x48,%rsp
    632d:	4c 89 f8             	mov    %r15,%rax
    6330:	5b                   	pop    %rbx
    6331:	5d                   	pop    %rbp
    6332:	41 5c                	pop    %r12
    6334:	41 5d                	pop    %r13
    6336:	41 5e                	pop    %r14
    6338:	41 5f                	pop    %r15
    633a:	c3                   	retq   
    633b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    6340:	49 63 c7             	movslq %r15d,%rax
    6343:	4d 8d 2c c6          	lea    (%r14,%rax,8),%r13
    6347:	eb b5                	jmp    62fe <jerasure_dumb_bitmatrix_to_schedule+0x13e>
  for (i = 0; i < m*w; i++) {
    6349:	49 89 c6             	mov    %rax,%r14
    634c:	eb c8                	jmp    6316 <jerasure_dumb_bitmatrix_to_schedule+0x156>
    634e:	66 90                	xchg   %ax,%ax

0000000000006350 <jerasure_smart_bitmatrix_to_schedule>:

int **jerasure_smart_bitmatrix_to_schedule(int k, int m, int w, int *bitmatrix)
{
    6350:	f3 0f 1e fa          	endbr64 
    6354:	41 57                	push   %r15
    6356:	41 56                	push   %r14
    6358:	41 55                	push   %r13
    635a:	41 89 fd             	mov    %edi,%r13d
    635d:	41 54                	push   %r12
    635f:	41 89 d4             	mov    %edx,%r12d
    6362:	55                   	push   %rbp
    6363:	53                   	push   %rbx
    6364:	89 f3                	mov    %esi,%ebx
  jerasure_print_bitmatrix(bitmatrix, m*w, k*w, w); */

  operations = talloc(int *, k*m*w*w+1);
  op = 0;
  
  diff = talloc(int, m*w);
    6366:	41 0f af dc          	imul   %r12d,%ebx
{
    636a:	48 81 ec 88 00 00 00 	sub    $0x88,%rsp
    6371:	89 7c 24 38          	mov    %edi,0x38(%rsp)
  operations = talloc(int *, k*m*w*w+1);
    6375:	0f af fe             	imul   %esi,%edi
  diff = talloc(int, m*w);
    6378:	48 63 eb             	movslq %ebx,%rbp
    637b:	48 c1 e5 02          	shl    $0x2,%rbp
  operations = talloc(int *, k*m*w*w+1);
    637f:	0f af fa             	imul   %edx,%edi
{
    6382:	48 89 4c 24 40       	mov    %rcx,0x40(%rsp)
  operations = talloc(int *, k*m*w*w+1);
    6387:	0f af fa             	imul   %edx,%edi
    638a:	ff c7                	inc    %edi
    638c:	48 63 ff             	movslq %edi,%rdi
    638f:	48 c1 e7 03          	shl    $0x3,%rdi
    6393:	e8 68 b0 ff ff       	callq  1400 <malloc@plt>
  diff = talloc(int, m*w);
    6398:	48 89 ef             	mov    %rbp,%rdi
  operations = talloc(int *, k*m*w*w+1);
    639b:	49 89 c7             	mov    %rax,%r15
  diff = talloc(int, m*w);
    639e:	e8 5d b0 ff ff       	callq  1400 <malloc@plt>
  from = talloc(int, m*w);
    63a3:	48 89 ef             	mov    %rbp,%rdi
  diff = talloc(int, m*w);
    63a6:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  from = talloc(int, m*w);
    63ab:	e8 50 b0 ff ff       	callq  1400 <malloc@plt>
  flink = talloc(int, m*w);
    63b0:	48 89 ef             	mov    %rbp,%rdi
  from = talloc(int, m*w);
    63b3:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  flink = talloc(int, m*w);
    63b8:	e8 43 b0 ff ff       	callq  1400 <malloc@plt>
  blink = talloc(int, m*w);
    63bd:	48 89 ef             	mov    %rbp,%rdi
  flink = talloc(int, m*w);
    63c0:	49 89 c6             	mov    %rax,%r14
  blink = talloc(int, m*w);
    63c3:	e8 38 b0 ff ff       	callq  1400 <malloc@plt>

  ptr = bitmatrix;

  bestdiff = k*w+1;
    63c8:	45 89 ea             	mov    %r13d,%r10d
    63cb:	45 0f af d4          	imul   %r12d,%r10d
  blink = talloc(int, m*w);
    63cf:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
  bestdiff = k*w+1;
    63d4:	45 8d 4a 01          	lea    0x1(%r10),%r9d
    63d8:	44 89 4c 24 70       	mov    %r9d,0x70(%rsp)
  top = 0;
  for (i = 0; i < m*w; i++) {
    63dd:	85 db                	test   %ebx,%ebx
    63df:	0f 8e 95 00 00 00    	jle    647a <jerasure_smart_bitmatrix_to_schedule+0x12a>
    63e5:	48 89 c7             	mov    %rax,%rdi
    63e8:	41 8d 42 ff          	lea    -0x1(%r10),%eax
    63ec:	48 89 6c 24 08       	mov    %rbp,0x8(%rsp)
    63f1:	4c 8d 1c 85 04 00 00 	lea    0x4(,%rax,4),%r11
    63f8:	00 
    63f9:	48 8b 6c 24 68       	mov    0x68(%rsp),%rbp
  ptr = bitmatrix;
    63fe:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    6403:	4c 8b 6c 24 18       	mov    0x18(%rsp),%r13
    6408:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
    640d:	ff cb                	dec    %ebx
  for (i = 0; i < m*w; i++) {
    640f:	31 f6                	xor    %esi,%esi
    6411:	49 89 ff             	mov    %rdi,%r15
    6414:	0f 1f 40 00          	nopl   0x0(%rax)
    6418:	89 f7                	mov    %esi,%edi
    641a:	41 89 f0             	mov    %esi,%r8d
    no = 0;
    for (j = 0; j < k*w; j++) {
    641d:	4a 8d 0c 18          	lea    (%rax,%r11,1),%rcx
    no = 0;
    6421:	31 d2                	xor    %edx,%edx
    for (j = 0; j < k*w; j++) {
    6423:	45 85 d2             	test   %r10d,%r10d
    6426:	7e 13                	jle    643b <jerasure_smart_bitmatrix_to_schedule+0xeb>
    6428:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    642f:	00 
      no += *ptr;
    6430:	03 10                	add    (%rax),%edx
      ptr++;
    6432:	48 83 c0 04          	add    $0x4,%rax
    for (j = 0; j < k*w; j++) {
    6436:	48 39 c8             	cmp    %rcx,%rax
    6439:	75 f5                	jne    6430 <jerasure_smart_bitmatrix_to_schedule+0xe0>
    }
    diff[i] = no;
    from[i] = -1;
    flink[i] = i+1;
    643b:	8d 4f 01             	lea    0x1(%rdi),%ecx
    blink[i] = i-1;
    643e:	ff cf                	dec    %edi
    diff[i] = no;
    6440:	89 54 b5 00          	mov    %edx,0x0(%rbp,%rsi,4)
    from[i] = -1;
    6444:	41 c7 44 b5 00 ff ff 	movl   $0xffffffff,0x0(%r13,%rsi,4)
    644b:	ff ff 
    flink[i] = i+1;
    644d:	41 89 0c b6          	mov    %ecx,(%r14,%rsi,4)
    blink[i] = i-1;
    6451:	41 89 3c b7          	mov    %edi,(%r15,%rsi,4)
    if (no < bestdiff) {
    6455:	41 39 d1             	cmp    %edx,%r9d
    6458:	7e 08                	jle    6462 <jerasure_smart_bitmatrix_to_schedule+0x112>
    645a:	44 89 44 24 14       	mov    %r8d,0x14(%rsp)
    645f:	41 89 d1             	mov    %edx,%r9d
  for (i = 0; i < m*w; i++) {
    6462:	48 8d 56 01          	lea    0x1(%rsi),%rdx
    6466:	48 39 f3             	cmp    %rsi,%rbx
    6469:	74 05                	je     6470 <jerasure_smart_bitmatrix_to_schedule+0x120>
    646b:	48 89 d6             	mov    %rdx,%rsi
    646e:	eb a8                	jmp    6418 <jerasure_smart_bitmatrix_to_schedule+0xc8>
    6470:	48 8b 6c 24 08       	mov    0x8(%rsp),%rbp
    6475:	4c 8b 7c 24 20       	mov    0x20(%rsp),%r15
      bestdiff = no;
      bestrow = i;
    }
  }

  flink[m*w-1] = -1;
    647a:	41 8d 42 ff          	lea    -0x1(%r10),%eax
    647e:	41 c7 44 2e fc ff ff 	movl   $0xffffffff,-0x4(%r14,%rbp,1)
    6485:	ff ff 
  top = 0;
    6487:	c7 44 24 5c 00 00 00 	movl   $0x0,0x5c(%rsp)
    648e:	00 
    648f:	89 44 24 58          	mov    %eax,0x58(%rsp)
  op = 0;
    6493:	45 31 c9             	xor    %r9d,%r9d
    6496:	4c 89 7c 24 60       	mov    %r15,0x60(%rsp)
    649b:	45 89 cf             	mov    %r9d,%r15d
    649e:	4d 89 f1             	mov    %r14,%r9
    64a1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  
  while (top != -1) {
    row = bestrow;
    /* printf("Doing row %d - %d from %d\n", row, diff[row], from[row]);  */

    if (blink[row] == -1) {
    64a8:	48 63 54 24 14       	movslq 0x14(%rsp),%rdx
    64ad:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    64b2:	8b 0c 90             	mov    (%rax,%rdx,4),%ecx
    64b5:	49 63 04 91          	movslq (%r9,%rdx,4),%rax
    64b9:	83 f9 ff             	cmp    $0xffffffff,%ecx
    64bc:	0f 84 92 03 00 00    	je     6854 <jerasure_smart_bitmatrix_to_schedule+0x504>
      top = flink[row];
      if (top != -1) blink[top] = -1;
    } else {
      flink[blink[row]] = flink[row];
    64c2:	48 63 f9             	movslq %ecx,%rdi
    64c5:	41 89 04 b9          	mov    %eax,(%r9,%rdi,4)
      if (flink[row] != -1) {
    64c9:	83 f8 ff             	cmp    $0xffffffff,%eax
    64cc:	74 08                	je     64d6 <jerasure_smart_bitmatrix_to_schedule+0x186>
        blink[flink[row]] = blink[row];
    64ce:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
    64d3:	89 0c 86             	mov    %ecx,(%rsi,%rax,4)
      }
    }

    ptr = bitmatrix + row*k*w;
    64d6:	8b 44 24 38          	mov    0x38(%rsp),%eax
    64da:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
    64df:	0f af 44 24 14       	imul   0x14(%rsp),%eax
    64e4:	4d 63 ef             	movslq %r15d,%r13
    64e7:	4e 8d 34 ed 00 00 00 	lea    0x0(,%r13,8),%r14
    64ee:	00 
    64ef:	41 0f af c4          	imul   %r12d,%eax
    64f3:	48 98                	cltq   
    64f5:	48 8d 1c 86          	lea    (%rsi,%rax,4),%rbx
    if (from[row] == -1) {
    64f9:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    64fe:	48 8b 74 24 60       	mov    0x60(%rsp),%rsi
    6503:	8b 2c 90             	mov    (%rax,%rdx,4),%ebp
    6506:	4e 8d 04 36          	lea    (%rsi,%r14,1),%r8
    650a:	83 fd ff             	cmp    $0xffffffff,%ebp
    650d:	0f 85 9c 01 00 00    	jne    66af <jerasure_smart_bitmatrix_to_schedule+0x35f>
      optodo = 0;
      for (j = 0; j < k*w; j++) {
    6513:	45 85 d2             	test   %r10d,%r10d
    6516:	0f 8e cc 00 00 00    	jle    65e8 <jerasure_smart_bitmatrix_to_schedule+0x298>
      optodo = 0;
    651c:	31 d2                	xor    %edx,%edx
    651e:	4c 89 4c 24 48       	mov    %r9,0x48(%rsp)
    6523:	44 89 54 24 3c       	mov    %r10d,0x3c(%rsp)
    6528:	45 89 e1             	mov    %r12d,%r9d
    652b:	44 8b 5c 24 58       	mov    0x58(%rsp),%r11d
    6530:	49 89 dc             	mov    %rbx,%r12
      for (j = 0; j < k*w; j++) {
    6533:	31 ed                	xor    %ebp,%ebp
    6535:	45 89 fe             	mov    %r15d,%r14d
    6538:	89 d3                	mov    %edx,%ebx
    653a:	eb 17                	jmp    6553 <jerasure_smart_bitmatrix_to_schedule+0x203>
    653c:	0f 1f 40 00          	nopl   0x0(%rax)
    6540:	48 8d 45 01          	lea    0x1(%rbp),%rax
    6544:	49 39 eb             	cmp    %rbp,%r11
    6547:	0f 84 88 00 00 00    	je     65d5 <jerasure_smart_bitmatrix_to_schedule+0x285>
    654d:	48 89 c5             	mov    %rax,%rbp
    6550:	4d 63 ee             	movslq %r14d,%r13
        if (ptr[j]) {
    6553:	41 8b 04 ac          	mov    (%r12,%rbp,4),%eax
    6557:	49 c1 e5 03          	shl    $0x3,%r13
    655b:	4e 8d 04 2e          	lea    (%rsi,%r13,1),%r8
    655f:	85 c0                	test   %eax,%eax
    6561:	74 dd                	je     6540 <jerasure_smart_bitmatrix_to_schedule+0x1f0>
          operations[op] = talloc(int, 5);
    6563:	bf 14 00 00 00       	mov    $0x14,%edi
    6568:	4c 89 44 24 30       	mov    %r8,0x30(%rsp)
    656d:	44 89 4c 24 28       	mov    %r9d,0x28(%rsp)
    6572:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
    6577:	4c 89 5c 24 08       	mov    %r11,0x8(%rsp)
    657c:	e8 7f ae ff ff       	callq  1400 <malloc@plt>
    6581:	4c 8b 44 24 30       	mov    0x30(%rsp),%r8
          operations[op][4] = optodo;
    6586:	89 58 10             	mov    %ebx,0x10(%rax)
          operations[op] = talloc(int, 5);
    6589:	48 89 c7             	mov    %rax,%rdi
    658c:	49 89 00             	mov    %rax,(%r8)
          operations[op][0] = j/w;
    658f:	44 8b 4c 24 28       	mov    0x28(%rsp),%r9d
    6594:	89 e8                	mov    %ebp,%eax
    6596:	99                   	cltd   
    6597:	41 f7 f9             	idiv   %r9d
    659a:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
          operations[op][1] = j%w;
          operations[op][2] = k+row/w;
          operations[op][3] = row%w;
          optodo = 1;
    659f:	4c 8b 5c 24 08       	mov    0x8(%rsp),%r11
          op++;
    65a4:	41 ff c6             	inc    %r14d
    65a7:	4e 8d 44 2e 08       	lea    0x8(%rsi,%r13,1),%r8
          optodo = 1;
    65ac:	bb 01 00 00 00       	mov    $0x1,%ebx
          operations[op][0] = j/w;
    65b1:	89 07                	mov    %eax,(%rdi)
          operations[op][1] = j%w;
    65b3:	8b 44 24 14          	mov    0x14(%rsp),%eax
    65b7:	89 57 04             	mov    %edx,0x4(%rdi)
          operations[op][2] = k+row/w;
    65ba:	99                   	cltd   
    65bb:	41 f7 f9             	idiv   %r9d
    65be:	03 44 24 38          	add    0x38(%rsp),%eax
    65c2:	89 47 08             	mov    %eax,0x8(%rdi)
          operations[op][3] = row%w;
    65c5:	89 57 0c             	mov    %edx,0xc(%rdi)
      for (j = 0; j < k*w; j++) {
    65c8:	48 8d 45 01          	lea    0x1(%rbp),%rax
    65cc:	49 39 eb             	cmp    %rbp,%r11
    65cf:	0f 85 78 ff ff ff    	jne    654d <jerasure_smart_bitmatrix_to_schedule+0x1fd>
    65d5:	4c 89 e3             	mov    %r12,%rbx
    65d8:	44 8b 54 24 3c       	mov    0x3c(%rsp),%r10d
    65dd:	45 89 cc             	mov    %r9d,%r12d
    65e0:	4c 8b 4c 24 48       	mov    0x48(%rsp),%r9
    65e5:	45 89 f7             	mov    %r14d,%r15d
          op++;
        }
      }
    }
    bestdiff = k*w+1;
    for (i = top; i != -1; i = flink[i]) {
    65e8:	44 8b 5c 24 5c       	mov    0x5c(%rsp),%r11d
    65ed:	41 83 fb ff          	cmp    $0xffffffff,%r11d
    65f1:	0f 84 fd 01 00 00    	je     67f4 <jerasure_smart_bitmatrix_to_schedule+0x4a4>
    65f7:	44 8b 44 24 58       	mov    0x58(%rsp),%r8d
    65fc:	44 89 7c 24 20       	mov    %r15d,0x20(%rsp)
    6601:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
    bestdiff = k*w+1;
    6606:	8b 6c 24 70          	mov    0x70(%rsp),%ebp
    for (i = top; i != -1; i = flink[i]) {
    660a:	44 8b 6c 24 14       	mov    0x14(%rsp),%r13d
    660f:	4c 8b 74 24 68       	mov    0x68(%rsp),%r14
    6614:	8b 74 24 38          	mov    0x38(%rsp),%esi
    6618:	4c 8b 7c 24 40       	mov    0x40(%rsp),%r15
    661d:	0f 1f 00             	nopl   (%rax)
      no = 1;
      b1 = bitmatrix + i*k*w;
    6620:	89 f0                	mov    %esi,%eax
    6622:	41 0f af c3          	imul   %r11d,%eax
    6626:	41 0f af c4          	imul   %r12d,%eax
    662a:	48 98                	cltq   
      for (j = 0; j < k*w; j++) no += (ptr[j] ^ b1[j]);
    662c:	45 85 d2             	test   %r10d,%r10d
    662f:	7e 77                	jle    66a8 <jerasure_smart_bitmatrix_to_schedule+0x358>
      no = 1;
    6631:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
    6636:	49 8d 3c 87          	lea    (%r15,%rax,4),%rdi
    663a:	b9 01 00 00 00       	mov    $0x1,%ecx
      for (j = 0; j < k*w; j++) no += (ptr[j] ^ b1[j]);
    663f:	31 c0                	xor    %eax,%eax
    6641:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    6648:	8b 14 83             	mov    (%rbx,%rax,4),%edx
    664b:	33 14 87             	xor    (%rdi,%rax,4),%edx
    664e:	01 d1                	add    %edx,%ecx
    6650:	48 89 c2             	mov    %rax,%rdx
    6653:	48 ff c0             	inc    %rax
    6656:	49 39 d0             	cmp    %rdx,%r8
    6659:	75 ed                	jne    6648 <jerasure_smart_bitmatrix_to_schedule+0x2f8>
    665b:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
      if (no < diff[i]) {
    6660:	49 63 d3             	movslq %r11d,%rdx
    6663:	49 8d 3c 96          	lea    (%r14,%rdx,4),%rdi
    6667:	8b 07                	mov    (%rdi),%eax
    6669:	39 c8                	cmp    %ecx,%eax
    666b:	7e 12                	jle    667f <jerasure_smart_bitmatrix_to_schedule+0x32f>
        from[i] = row;
    666d:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    6672:	44 8b 44 24 14       	mov    0x14(%rsp),%r8d
        diff[i] = no;
    6677:	89 0f                	mov    %ecx,(%rdi)
        from[i] = row;
    6679:	44 89 04 90          	mov    %r8d,(%rax,%rdx,4)
        diff[i] = no;
    667d:	89 c8                	mov    %ecx,%eax
      }
      if (diff[i] < bestdiff) {
    667f:	39 c5                	cmp    %eax,%ebp
    6681:	7e 05                	jle    6688 <jerasure_smart_bitmatrix_to_schedule+0x338>
    6683:	89 c5                	mov    %eax,%ebp
    6685:	45 89 dd             	mov    %r11d,%r13d
    for (i = top; i != -1; i = flink[i]) {
    6688:	45 8b 1c 91          	mov    (%r9,%rdx,4),%r11d
    668c:	41 83 fb ff          	cmp    $0xffffffff,%r11d
    6690:	75 8e                	jne    6620 <jerasure_smart_bitmatrix_to_schedule+0x2d0>
    6692:	44 89 6c 24 14       	mov    %r13d,0x14(%rsp)
    6697:	44 8b 7c 24 20       	mov    0x20(%rsp),%r15d
    669c:	e9 07 fe ff ff       	jmpq   64a8 <jerasure_smart_bitmatrix_to_schedule+0x158>
    66a1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
      no = 1;
    66a8:	b9 01 00 00 00       	mov    $0x1,%ecx
    66ad:	eb b1                	jmp    6660 <jerasure_smart_bitmatrix_to_schedule+0x310>
      operations[op] = talloc(int, 5);
    66af:	bf 14 00 00 00       	mov    $0x14,%edi
    66b4:	4c 89 44 24 28       	mov    %r8,0x28(%rsp)
    66b9:	4c 89 4c 24 20       	mov    %r9,0x20(%rsp)
    66be:	44 89 54 24 08       	mov    %r10d,0x8(%rsp)
    66c3:	e8 38 ad ff ff       	callq  1400 <malloc@plt>
    66c8:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
      operations[op][4] = 0;
    66cd:	c7 40 10 00 00 00 00 	movl   $0x0,0x10(%rax)
      operations[op] = talloc(int, 5);
    66d4:	48 89 c1             	mov    %rax,%rcx
    66d7:	49 89 00             	mov    %rax,(%r8)
      operations[op][0] = k+from[row]/w;
    66da:	89 e8                	mov    %ebp,%eax
    66dc:	99                   	cltd   
    66dd:	41 f7 fc             	idiv   %r12d
    66e0:	8b 74 24 38          	mov    0x38(%rsp),%esi
      for (j = 0; j < k*w; j++) {
    66e4:	44 8b 54 24 08       	mov    0x8(%rsp),%r10d
    66e9:	4c 8b 4c 24 20       	mov    0x20(%rsp),%r9
      b1 = bitmatrix + from[row]*k*w;
    66ee:	0f af ee             	imul   %esi,%ebp
      op++;
    66f1:	41 ff c7             	inc    %r15d
      b1 = bitmatrix + from[row]*k*w;
    66f4:	41 0f af ec          	imul   %r12d,%ebp
    66f8:	48 63 ed             	movslq %ebp,%rbp
      operations[op][0] = k+from[row]/w;
    66fb:	01 f0                	add    %esi,%eax
    66fd:	89 01                	mov    %eax,(%rcx)
      operations[op][1] = from[row]%w;
    66ff:	8b 44 24 14          	mov    0x14(%rsp),%eax
    6703:	89 51 04             	mov    %edx,0x4(%rcx)
      operations[op][2] = k+row/w;
    6706:	99                   	cltd   
    6707:	41 f7 fc             	idiv   %r12d
    670a:	01 f0                	add    %esi,%eax
      for (j = 0; j < k*w; j++) {
    670c:	45 85 d2             	test   %r10d,%r10d
    670f:	89 54 24 48          	mov    %edx,0x48(%rsp)
      operations[op][2] = k+row/w;
    6713:	89 44 24 3c          	mov    %eax,0x3c(%rsp)
    6717:	89 41 08             	mov    %eax,0x8(%rcx)
      operations[op][3] = row%w;
    671a:	89 51 0c             	mov    %edx,0xc(%rcx)
      for (j = 0; j < k*w; j++) {
    671d:	0f 8e 5a 01 00 00    	jle    687d <jerasure_smart_bitmatrix_to_schedule+0x52d>
    6723:	8b 44 24 58          	mov    0x58(%rsp),%eax
    6727:	44 89 64 24 30       	mov    %r12d,0x30(%rsp)
    672c:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    6731:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    6736:	44 89 54 24 74       	mov    %r10d,0x74(%rsp)
    673b:	4c 8d 34 a8          	lea    (%rax,%rbp,4),%r14
    673f:	48 89 d8             	mov    %rbx,%rax
    6742:	4d 89 f4             	mov    %r14,%r12
    6745:	44 89 fb             	mov    %r15d,%ebx
    6748:	4c 89 4c 24 78       	mov    %r9,0x78(%rsp)
    674d:	4c 8b 74 24 60       	mov    0x60(%rsp),%r14
    6752:	31 ed                	xor    %ebp,%ebp
    6754:	49 89 c7             	mov    %rax,%r15
    6757:	eb 0a                	jmp    6763 <jerasure_smart_bitmatrix_to_schedule+0x413>
    6759:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    6760:	48 89 c5             	mov    %rax,%rbp
    6763:	4c 63 eb             	movslq %ebx,%r13
        if (ptr[j] ^ b1[j]) {
    6766:	41 8b 04 ac          	mov    (%r12,%rbp,4),%eax
    676a:	49 c1 e5 03          	shl    $0x3,%r13
    676e:	89 6c 24 08          	mov    %ebp,0x8(%rsp)
    6772:	4f 8d 04 2e          	lea    (%r14,%r13,1),%r8
    6776:	41 39 04 af          	cmp    %eax,(%r15,%rbp,4)
    677a:	74 46                	je     67c2 <jerasure_smart_bitmatrix_to_schedule+0x472>
          operations[op] = talloc(int, 5);
    677c:	bf 14 00 00 00       	mov    $0x14,%edi
    6781:	4c 89 44 24 28       	mov    %r8,0x28(%rsp)
    6786:	e8 75 ac ff ff       	callq  1400 <malloc@plt>
    678b:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
    6790:	8b 54 24 08          	mov    0x8(%rsp),%edx
          operations[op][4] = 1;
    6794:	c7 40 10 01 00 00 00 	movl   $0x1,0x10(%rax)
          operations[op] = talloc(int, 5);
    679b:	48 89 c1             	mov    %rax,%rcx
    679e:	49 89 00             	mov    %rax,(%r8)
          operations[op][0] = j/w;
    67a1:	89 d0                	mov    %edx,%eax
    67a3:	99                   	cltd   
    67a4:	f7 7c 24 30          	idivl  0x30(%rsp)
          op++;
    67a8:	ff c3                	inc    %ebx
    67aa:	4f 8d 44 2e 08       	lea    0x8(%r14,%r13,1),%r8
          operations[op][0] = j/w;
    67af:	89 01                	mov    %eax,(%rcx)
          operations[op][2] = k+row/w;
    67b1:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
          operations[op][1] = j%w;
    67b5:	89 51 04             	mov    %edx,0x4(%rcx)
          operations[op][2] = k+row/w;
    67b8:	89 41 08             	mov    %eax,0x8(%rcx)
          operations[op][3] = row%w;
    67bb:	8b 44 24 48          	mov    0x48(%rsp),%eax
    67bf:	89 41 0c             	mov    %eax,0xc(%rcx)
      for (j = 0; j < k*w; j++) {
    67c2:	48 8d 45 01          	lea    0x1(%rbp),%rax
    67c6:	48 39 6c 24 20       	cmp    %rbp,0x20(%rsp)
    67cb:	75 93                	jne    6760 <jerasure_smart_bitmatrix_to_schedule+0x410>
    for (i = top; i != -1; i = flink[i]) {
    67cd:	44 8b 5c 24 5c       	mov    0x5c(%rsp),%r11d
    67d2:	4c 89 f8             	mov    %r15,%rax
    67d5:	44 8b 54 24 74       	mov    0x74(%rsp),%r10d
    67da:	41 89 df             	mov    %ebx,%r15d
    67dd:	4c 8b 4c 24 78       	mov    0x78(%rsp),%r9
    67e2:	44 8b 64 24 30       	mov    0x30(%rsp),%r12d
    67e7:	48 89 c3             	mov    %rax,%rbx
    67ea:	41 83 fb ff          	cmp    $0xffffffff,%r11d
    67ee:	0f 85 03 fe ff ff    	jne    65f7 <jerasure_smart_bitmatrix_to_schedule+0x2a7>
        bestrow = i;
      }
    }
  }
  
  operations[op] = talloc(int, 5);
    67f4:	bf 14 00 00 00       	mov    $0x14,%edi
    67f9:	4d 89 ce             	mov    %r9,%r14
    67fc:	4c 8b 7c 24 60       	mov    0x60(%rsp),%r15
    6801:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
    6806:	e8 f5 ab ff ff       	callq  1400 <malloc@plt>
    680b:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
  operations[op][0] = -1;
    6810:	c7 00 ff ff ff ff    	movl   $0xffffffff,(%rax)
  operations[op] = talloc(int, 5);
    6816:	49 89 00             	mov    %rax,(%r8)
  free(from);
    6819:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    681e:	e8 5d aa ff ff       	callq  1280 <free@plt>
  free(diff);
    6823:	48 8b 7c 24 68       	mov    0x68(%rsp),%rdi
    6828:	e8 53 aa ff ff       	callq  1280 <free@plt>
  free(blink);
    682d:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
    6832:	e8 49 aa ff ff       	callq  1280 <free@plt>
  free(flink);
    6837:	4c 89 f7             	mov    %r14,%rdi
    683a:	e8 41 aa ff ff       	callq  1280 <free@plt>

  return operations;
}
    683f:	48 81 c4 88 00 00 00 	add    $0x88,%rsp
    6846:	5b                   	pop    %rbx
    6847:	5d                   	pop    %rbp
    6848:	41 5c                	pop    %r12
    684a:	41 5d                	pop    %r13
    684c:	41 5e                	pop    %r14
    684e:	4c 89 f8             	mov    %r15,%rax
    6851:	41 5f                	pop    %r15
    6853:	c3                   	retq   
      if (top != -1) blink[top] = -1;
    6854:	c7 44 24 5c ff ff ff 	movl   $0xffffffff,0x5c(%rsp)
    685b:	ff 
    685c:	83 f8 ff             	cmp    $0xffffffff,%eax
    685f:	0f 84 71 fc ff ff    	je     64d6 <jerasure_smart_bitmatrix_to_schedule+0x186>
    6865:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
    686a:	48 63 c8             	movslq %eax,%rcx
    686d:	c7 04 8e ff ff ff ff 	movl   $0xffffffff,(%rsi,%rcx,4)
    6874:	89 44 24 5c          	mov    %eax,0x5c(%rsp)
    6878:	e9 59 fc ff ff       	jmpq   64d6 <jerasure_smart_bitmatrix_to_schedule+0x186>
    687d:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    6882:	4e 8d 44 30 08       	lea    0x8(%rax,%r14,1),%r8
    6887:	e9 5c fd ff ff       	jmpq   65e8 <jerasure_smart_bitmatrix_to_schedule+0x298>
    688c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000006890 <jerasure_generate_decoding_schedule>:
{
    6890:	41 57                	push   %r15
    6892:	41 56                	push   %r14
    6894:	4d 89 c6             	mov    %r8,%r14
    6897:	41 55                	push   %r13
    6899:	41 54                	push   %r12
    689b:	41 89 f4             	mov    %esi,%r12d
    689e:	55                   	push   %rbp
    689f:	53                   	push   %rbx
    68a0:	89 d3                	mov    %edx,%ebx
    68a2:	48 81 ec b8 00 00 00 	sub    $0xb8,%rsp
  for (i = 0; erasures[i] != -1; i++) {
    68a9:	41 8b 00             	mov    (%r8),%eax
{
    68ac:	89 7c 24 24          	mov    %edi,0x24(%rsp)
    68b0:	48 89 4c 24 78       	mov    %rcx,0x78(%rsp)
    68b5:	44 89 8c 24 ac 00 00 	mov    %r9d,0xac(%rsp)
    68bc:	00 
  for (i = 0; erasures[i] != -1; i++) {
    68bd:	83 f8 ff             	cmp    $0xffffffff,%eax
    68c0:	0f 84 8c 06 00 00    	je     6f52 <jerasure_generate_decoding_schedule+0x6c2>
  cdf = 0;
    68c6:	c7 44 24 08 00 00 00 	movl   $0x0,0x8(%rsp)
    68cd:	00 
    68ce:	49 8d 50 04          	lea    0x4(%r8),%rdx
  ddf = 0;
    68d2:	31 ed                	xor    %ebp,%ebp
    68d4:	eb 0d                	jmp    68e3 <jerasure_generate_decoding_schedule+0x53>
  for (i = 0; erasures[i] != -1; i++) {
    68d6:	8b 02                	mov    (%rdx),%eax
    68d8:	48 83 c2 04          	add    $0x4,%rdx
    if (erasures[i] < k) ddf++; else cdf++;
    68dc:	ff c5                	inc    %ebp
  for (i = 0; erasures[i] != -1; i++) {
    68de:	83 f8 ff             	cmp    $0xffffffff,%eax
    68e1:	74 15                	je     68f8 <jerasure_generate_decoding_schedule+0x68>
    if (erasures[i] < k) ddf++; else cdf++;
    68e3:	39 44 24 24          	cmp    %eax,0x24(%rsp)
    68e7:	7f ed                	jg     68d6 <jerasure_generate_decoding_schedule+0x46>
  for (i = 0; erasures[i] != -1; i++) {
    68e9:	8b 02                	mov    (%rdx),%eax
    68eb:	48 83 c2 04          	add    $0x4,%rdx
    if (erasures[i] < k) ddf++; else cdf++;
    68ef:	ff 44 24 08          	incl   0x8(%rsp)
  for (i = 0; erasures[i] != -1; i++) {
    68f3:	83 f8 ff             	cmp    $0xffffffff,%eax
    68f6:	75 eb                	jne    68e3 <jerasure_generate_decoding_schedule+0x53>
  row_ids = talloc(int, k+m);
    68f8:	8b 7c 24 24          	mov    0x24(%rsp),%edi
    68fc:	46 8d 2c 27          	lea    (%rdi,%r12,1),%r13d
    6900:	4d 63 fd             	movslq %r13d,%r15
    6903:	49 c1 e7 02          	shl    $0x2,%r15
    6907:	4c 89 ff             	mov    %r15,%rdi
    690a:	e8 f1 aa ff ff       	callq  1400 <malloc@plt>
  ind_to_row = talloc(int, k+m);
    690f:	4c 89 ff             	mov    %r15,%rdi
  row_ids = talloc(int, k+m);
    6912:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  ind_to_row = talloc(int, k+m);
    6917:	e8 e4 aa ff ff       	callq  1400 <malloc@plt>
  erased = jerasure_erasures_to_erased(k, m, erasures);
    691c:	4c 89 f2             	mov    %r14,%rdx
    691f:	44 8b 74 24 24       	mov    0x24(%rsp),%r14d
    6924:	44 89 e6             	mov    %r12d,%esi
    6927:	44 89 f7             	mov    %r14d,%edi
  ind_to_row = talloc(int, k+m);
    692a:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    692f:	49 89 c7             	mov    %rax,%r15
  erased = jerasure_erasures_to_erased(k, m, erasures);
    6932:	e8 99 db ff ff       	callq  44d0 <jerasure_erasures_to_erased>
    6937:	49 89 c4             	mov    %rax,%r12
  if (erased == NULL) return -1;
    693a:	48 85 c0             	test   %rax,%rax
    693d:	0f 84 45 04 00 00    	je     6d88 <jerasure_generate_decoding_schedule+0x4f8>
  for (i = 0; i < k; i++) {
    6943:	45 85 f6             	test   %r14d,%r14d
    6946:	0f 8e fd 05 00 00    	jle    6f49 <jerasure_generate_decoding_schedule+0x6b9>
    694c:	4c 8b 4c 24 10       	mov    0x10(%rsp),%r9
    6951:	44 89 f6             	mov    %r14d,%esi
    6954:	45 8d 56 ff          	lea    -0x1(%r14),%r10d
    6958:	44 89 f1             	mov    %r14d,%ecx
    695b:	31 d2                	xor    %edx,%edx
    695d:	4d 89 fb             	mov    %r15,%r11
    6960:	eb 14                	jmp    6976 <jerasure_generate_decoding_schedule+0xe6>
      row_ids[i] = i;
    6962:	41 89 14 91          	mov    %edx,(%r9,%rdx,4)
      ind_to_row[i] = i;
    6966:	41 89 14 93          	mov    %edx,(%r11,%rdx,4)
  for (i = 0; i < k; i++) {
    696a:	48 8d 42 01          	lea    0x1(%rdx),%rax
    696e:	4c 39 d2             	cmp    %r10,%rdx
    6971:	74 5e                	je     69d1 <jerasure_generate_decoding_schedule+0x141>
    6973:	48 89 c2             	mov    %rax,%rdx
    if (erased[i] == 0) {
    6976:	41 8b 04 94          	mov    (%r12,%rdx,4),%eax
    697a:	41 89 d0             	mov    %edx,%r8d
    697d:	85 c0                	test   %eax,%eax
    697f:	74 e1                	je     6962 <jerasure_generate_decoding_schedule+0xd2>
      while (erased[j]) j++;
    6981:	4c 63 f1             	movslq %ecx,%r14
    6984:	47 8b 3c b4          	mov    (%r12,%r14,4),%r15d
    6988:	8d 41 01             	lea    0x1(%rcx),%eax
    698b:	4a 8d 3c b5 00 00 00 	lea    0x0(,%r14,4),%rdi
    6992:	00 
    6993:	48 98                	cltq   
    6995:	45 85 ff             	test   %r15d,%r15d
    6998:	74 17                	je     69b1 <jerasure_generate_decoding_schedule+0x121>
    699a:	89 c1                	mov    %eax,%ecx
    699c:	48 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%rdi
    69a3:	00 
    69a4:	48 ff c0             	inc    %rax
    69a7:	45 8b 74 84 fc       	mov    -0x4(%r12,%rax,4),%r14d
    69ac:	45 85 f6             	test   %r14d,%r14d
    69af:	75 e9                	jne    699a <jerasure_generate_decoding_schedule+0x10a>
      row_ids[x] = i;
    69b1:	48 63 c6             	movslq %esi,%rax
      row_ids[i] = j;
    69b4:	41 89 0c 91          	mov    %ecx,(%r9,%rdx,4)
      ind_to_row[j] = i;
    69b8:	45 89 04 3b          	mov    %r8d,(%r11,%rdi,1)
      row_ids[x] = i;
    69bc:	45 89 04 81          	mov    %r8d,(%r9,%rax,4)
      ind_to_row[i] = x;
    69c0:	41 89 34 93          	mov    %esi,(%r11,%rdx,4)
      j++;
    69c4:	ff c1                	inc    %ecx
      x++;
    69c6:	ff c6                	inc    %esi
  for (i = 0; i < k; i++) {
    69c8:	48 8d 42 01          	lea    0x1(%rdx),%rax
    69cc:	4c 39 d2             	cmp    %r10,%rdx
    69cf:	75 a2                	jne    6973 <jerasure_generate_decoding_schedule+0xe3>
  for (i = k; i < k+m; i++) {
    69d1:	48 63 44 24 24       	movslq 0x24(%rsp),%rax
    69d6:	41 39 c5             	cmp    %eax,%r13d
    69d9:	7e 25                	jle    6a00 <jerasure_generate_decoding_schedule+0x170>
    if (erased[i]) {
    69db:	41 8b 3c 84          	mov    (%r12,%rax,4),%edi
    69df:	85 ff                	test   %edi,%edi
    69e1:	74 15                	je     69f8 <jerasure_generate_decoding_schedule+0x168>
      row_ids[x] = i;
    69e3:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    69e8:	48 63 d6             	movslq %esi,%rdx
    69eb:	89 04 97             	mov    %eax,(%rdi,%rdx,4)
      ind_to_row[i] = x;
    69ee:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    69f3:	89 34 87             	mov    %esi,(%rdi,%rax,4)
      x++;
    69f6:	ff c6                	inc    %esi
  for (i = k; i < k+m; i++) {
    69f8:	48 ff c0             	inc    %rax
    69fb:	41 39 c5             	cmp    %eax,%r13d
    69fe:	7f db                	jg     69db <jerasure_generate_decoding_schedule+0x14b>
  free(erased);
    6a00:	4c 89 e7             	mov    %r12,%rdi
    6a03:	e8 78 a8 ff ff       	callq  1280 <free@plt>
  real_decoding_matrix = talloc(int, k*w*(cdf+ddf)*w);
    6a08:	44 8b 74 24 24       	mov    0x24(%rsp),%r14d
    6a0d:	8b 44 24 08          	mov    0x8(%rsp),%eax
    6a11:	44 0f af f3          	imul   %ebx,%r14d
    6a15:	01 e8                	add    %ebp,%eax
    6a17:	89 84 24 a8 00 00 00 	mov    %eax,0xa8(%rsp)
    6a1e:	41 0f af c6          	imul   %r14d,%eax
    6a22:	89 c7                	mov    %eax,%edi
    6a24:	0f af fb             	imul   %ebx,%edi
    6a27:	48 63 ff             	movslq %edi,%rdi
    6a2a:	48 c1 e7 02          	shl    $0x2,%rdi
    6a2e:	e8 cd a9 ff ff       	callq  1400 <malloc@plt>
    6a33:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  if (ddf > 0) {
    6a38:	85 ed                	test   %ebp,%ebp
    6a3a:	0f 85 79 03 00 00    	jne    6db9 <jerasure_generate_decoding_schedule+0x529>
  for (x = 0; x < cdf; x++) {
    6a40:	8b 4c 24 08          	mov    0x8(%rsp),%ecx
    6a44:	85 c9                	test   %ecx,%ecx
    6a46:	0f 84 f9 02 00 00    	je     6d45 <jerasure_generate_decoding_schedule+0x4b5>
    memcpy(ptr, bitmatrix+drive*k*w*w, sizeof(int)*k*w*w);
    6a4c:	48 63 f3             	movslq %ebx,%rsi
    6a4f:	48 89 f7             	mov    %rsi,%rdi
    6a52:	48 0f af fe          	imul   %rsi,%rdi
    6a56:	48 63 54 24 24       	movslq 0x24(%rsp),%rdx
    ptr = real_decoding_matrix + k*w*w*(ddf+x);
    6a5b:	44 89 f0             	mov    %r14d,%eax
    memcpy(ptr, bitmatrix+drive*k*w*w, sizeof(int)*k*w*w);
    6a5e:	48 0f af fa          	imul   %rdx,%rdi
    ptr = real_decoding_matrix + k*w*w*(ddf+x);
    6a62:	0f af c3             	imul   %ebx,%eax
    memcpy(ptr, bitmatrix+drive*k*w*w, sizeof(int)*k*w*w);
    6a65:	49 89 d3             	mov    %rdx,%r11
    6a68:	48 8d 3c bd 00 00 00 	lea    0x0(,%rdi,4),%rdi
    6a6f:	00 
    6a70:	48 89 bc 24 80 00 00 	mov    %rdi,0x80(%rsp)
    6a77:	00 
          bzero(ptr+j*k*w+i*w, sizeof(int)*w);
    6a78:	48 8d 3c b5 00 00 00 	lea    0x0(,%rsi,4),%rdi
    6a7f:	00 
    6a80:	48 63 d5             	movslq %ebp,%rdx
    6a83:	48 89 bc 24 88 00 00 	mov    %rdi,0x88(%rsp)
    6a8a:	00 
    6a8b:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    6a90:	4c 01 da             	add    %r11,%rdx
    6a93:	48 8d 34 97          	lea    (%rdi,%rdx,4),%rsi
    6a97:	48 63 c8             	movslq %eax,%rcx
    6a9a:	0f af c5             	imul   %ebp,%eax
    6a9d:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
    6aa2:	48 8d 34 8d 00 00 00 	lea    0x0(,%rcx,4),%rsi
    6aa9:	00 
    6aaa:	48 89 b4 24 a0 00 00 	mov    %rsi,0xa0(%rsp)
    6ab1:	00 
    6ab2:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
    6ab7:	48 98                	cltq   
    6ab9:	48 8d 04 86          	lea    (%rsi,%rax,4),%rax
    6abd:	48 89 c6             	mov    %rax,%rsi
    6ac0:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    6ac5:	41 8d 46 ff          	lea    -0x1(%r14),%eax
    6ac9:	48 8d 44 86 04       	lea    0x4(%rsi,%rax,4),%rax
    6ace:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    6ad3:	8b 44 24 08          	mov    0x8(%rsp),%eax
    6ad7:	44 0f af db          	imul   %ebx,%r11d
    6adb:	ff c8                	dec    %eax
    6add:	48 01 c2             	add    %rax,%rdx
    6ae0:	48 8d 44 97 04       	lea    0x4(%rdi,%rdx,4),%rax
    6ae5:	48 89 84 24 98 00 00 	mov    %rax,0x98(%rsp)
    6aec:	00 
    6aed:	49 63 c3             	movslq %r11d,%rax
    6af0:	48 89 c5             	mov    %rax,%rbp
    6af3:	4c 8d 2c 85 00 00 00 	lea    0x0(,%rax,4),%r13
    6afa:	00 
    6afb:	8d 43 ff             	lea    -0x1(%rbx),%eax
    6afe:	48 89 84 24 90 00 00 	mov    %rax,0x90(%rsp)
    6b05:	00 
    6b06:	48 f7 d0             	not    %rax
    6b09:	48 c1 e0 02          	shl    $0x2,%rax
    6b0d:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
    drive = row_ids[x+ddf+k]-k;
    6b12:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
    6b17:	44 8b 7c 24 24       	mov    0x24(%rsp),%r15d
    6b1c:	8b 00                	mov    (%rax),%eax
    memcpy(ptr, bitmatrix+drive*k*w*w, sizeof(int)*k*w*w);
    6b1e:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
    drive = row_ids[x+ddf+k]-k;
    6b23:	44 29 f8             	sub    %r15d,%eax
    memcpy(ptr, bitmatrix+drive*k*w*w, sizeof(int)*k*w*w);
    6b26:	41 0f af c7          	imul   %r15d,%eax
    6b2a:	48 8b 94 24 80 00 00 	mov    0x80(%rsp),%rdx
    6b31:	00 
    6b32:	0f af c3             	imul   %ebx,%eax
    6b35:	0f af c3             	imul   %ebx,%eax
    6b38:	48 98                	cltq   
    6b3a:	48 8d 34 87          	lea    (%rdi,%rax,4),%rsi
    6b3e:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    6b43:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    6b48:	e8 73 a8 ff ff       	callq  13c0 <memcpy@plt>
    for (i = 0; i < k; i++) {
    6b4d:	45 85 ff             	test   %r15d,%r15d
    6b50:	0f 8e c4 01 00 00    	jle    6d1a <jerasure_generate_decoding_schedule+0x48a>
    6b56:	8b 44 24 24          	mov    0x24(%rsp),%eax
    6b5a:	31 ff                	xor    %edi,%edi
    6b5c:	ff c8                	dec    %eax
    6b5e:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    6b63:	31 c0                	xor    %eax,%eax
    6b65:	44 89 74 24 28       	mov    %r14d,0x28(%rsp)
    6b6a:	89 6c 24 38          	mov    %ebp,0x38(%rsp)
    6b6e:	4c 8b a4 24 88 00 00 	mov    0x88(%rsp),%r12
    6b75:	00 
    6b76:	41 89 fe             	mov    %edi,%r14d
    6b79:	48 89 c5             	mov    %rax,%rbp
    6b7c:	eb 11                	jmp    6b8f <jerasure_generate_decoding_schedule+0x2ff>
    6b7e:	48 8d 45 01          	lea    0x1(%rbp),%rax
    6b82:	41 01 de             	add    %ebx,%r14d
    6b85:	48 39 6c 24 18       	cmp    %rbp,0x18(%rsp)
    6b8a:	74 4a                	je     6bd6 <jerasure_generate_decoding_schedule+0x346>
    6b8c:	48 89 c5             	mov    %rax,%rbp
      if (row_ids[i] != i) {
    6b8f:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    6b94:	39 2c a8             	cmp    %ebp,(%rax,%rbp,4)
    6b97:	74 e5                	je     6b7e <jerasure_generate_decoding_schedule+0x2ee>
        for (j = 0; j < w; j++) {
    6b99:	85 db                	test   %ebx,%ebx
    6b9b:	7e e1                	jle    6b7e <jerasure_generate_decoding_schedule+0x2ee>
    6b9d:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
          bzero(ptr+j*k*w+i*w, sizeof(int)*w);
    6ba2:	49 63 c6             	movslq %r14d,%rax
    6ba5:	48 8d 3c 87          	lea    (%rdi,%rax,4),%rdi
        for (j = 0; j < w; j++) {
    6ba9:	45 31 ff             	xor    %r15d,%r15d
    6bac:	0f 1f 40 00          	nopl   0x0(%rax)
    6bb0:	4c 89 e2             	mov    %r12,%rdx
    6bb3:	31 f6                	xor    %esi,%esi
    6bb5:	e8 96 a7 ff ff       	callq  1350 <memset@plt>
    6bba:	48 89 c7             	mov    %rax,%rdi
    6bbd:	41 ff c7             	inc    %r15d
    6bc0:	4c 01 ef             	add    %r13,%rdi
    6bc3:	44 39 fb             	cmp    %r15d,%ebx
    6bc6:	75 e8                	jne    6bb0 <jerasure_generate_decoding_schedule+0x320>
    for (i = 0; i < k; i++) {
    6bc8:	48 8d 45 01          	lea    0x1(%rbp),%rax
    6bcc:	41 01 de             	add    %ebx,%r14d
    6bcf:	48 39 6c 24 18       	cmp    %rbp,0x18(%rsp)
    6bd4:	75 b6                	jne    6b8c <jerasure_generate_decoding_schedule+0x2fc>
    6bd6:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    6bdb:	44 8b 74 24 28       	mov    0x28(%rsp),%r14d
    6be0:	48 03 84 24 90 00 00 	add    0x90(%rsp),%rax
    6be7:	00 
    6be8:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    6bed:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    6bf2:	c7 44 24 28 00 00 00 	movl   $0x0,0x28(%rsp)
    6bf9:	00 
    6bfa:	48 83 c0 04          	add    $0x4,%rax
    6bfe:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    6c03:	8b 6c 24 38          	mov    0x38(%rsp),%ebp
    6c07:	4c 8b 64 24 48       	mov    0x48(%rsp),%r12
    6c0c:	45 31 ff             	xor    %r15d,%r15d
    6c0f:	eb 16                	jmp    6c27 <jerasure_generate_decoding_schedule+0x397>
    for (i = 0; i < k; i++) {
    6c11:	01 5c 24 28          	add    %ebx,0x28(%rsp)
    6c15:	49 8d 47 01          	lea    0x1(%r15),%rax
    6c19:	4c 39 7c 24 18       	cmp    %r15,0x18(%rsp)
    6c1e:	0f 84 f6 00 00 00    	je     6d1a <jerasure_generate_decoding_schedule+0x48a>
    6c24:	49 89 c7             	mov    %rax,%r15
      if (row_ids[i] != i) {
    6c27:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    6c2c:	46 39 3c b8          	cmp    %r15d,(%rax,%r15,4)
    6c30:	74 df                	je     6c11 <jerasure_generate_decoding_schedule+0x381>
        b1 = real_decoding_matrix+(ind_to_row[i]-k)*k*w*w;
    6c32:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    6c37:	42 8b 04 b8          	mov    (%rax,%r15,4),%eax
    6c3b:	41 89 c2             	mov    %eax,%r10d
    6c3e:	89 44 24 08          	mov    %eax,0x8(%rsp)
    6c42:	8b 44 24 24          	mov    0x24(%rsp),%eax
    6c46:	41 29 c2             	sub    %eax,%r10d
    6c49:	44 0f af d0          	imul   %eax,%r10d
    6c4d:	44 0f af d3          	imul   %ebx,%r10d
    6c51:	44 0f af d3          	imul   %ebx,%r10d
    6c55:	4d 63 d2             	movslq %r10d,%r10
        for (j = 0; j < w; j++) {
    6c58:	85 db                	test   %ebx,%ebx
    6c5a:	7e b5                	jle    6c11 <jerasure_generate_decoding_schedule+0x381>
    6c5c:	48 63 44 24 28       	movslq 0x28(%rsp),%rax
    6c61:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
    6c66:	4c 89 7c 24 38       	mov    %r15,0x38(%rsp)
    6c6b:	48 03 44 24 58       	add    0x58(%rsp),%rax
            if (bitmatrix[index+j*k*w+i*w+y]) {
    6c70:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
    6c75:	4c 8b 4c 24 30       	mov    0x30(%rsp),%r9
        for (j = 0; j < w; j++) {
    6c7a:	4c 8b 7c 24 68       	mov    0x68(%rsp),%r15
    6c7f:	4c 8d 04 87          	lea    (%rdi,%rax,4),%r8
    6c83:	45 31 db             	xor    %r11d,%r11d
    6c86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    6c8d:	00 00 00 
          for (y = 0; y < w; y++) {
    6c90:	4b 8d 0c 07          	lea    (%r15,%r8,1),%rcx
        for (j = 0; j < w; j++) {
    6c94:	31 ff                	xor    %edi,%edi
    6c96:	eb 13                	jmp    6cab <jerasure_generate_decoding_schedule+0x41b>
    6c98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    6c9f:	00 
          for (y = 0; y < w; y++) {
    6ca0:	48 83 c1 04          	add    $0x4,%rcx
    6ca4:	01 ef                	add    %ebp,%edi
    6ca6:	49 39 c8             	cmp    %rcx,%r8
    6ca9:	74 46                	je     6cf1 <jerasure_generate_decoding_schedule+0x461>
            if (bitmatrix[index+j*k*w+i*w+y]) {
    6cab:	8b 01                	mov    (%rcx),%eax
    6cad:	85 c0                	test   %eax,%eax
    6caf:	74 ef                	je     6ca0 <jerasure_generate_decoding_schedule+0x410>
              for (z = 0; z < k*w; z++) {
    6cb1:	45 85 f6             	test   %r14d,%r14d
    6cb4:	7e ea                	jle    6ca0 <jerasure_generate_decoding_schedule+0x410>
    6cb6:	48 63 c7             	movslq %edi,%rax
                b2[z] = b2[z] ^ b1[z+y*k*w];
    6cb9:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
    6cbe:	4c 01 d0             	add    %r10,%rax
    6cc1:	49 8d 14 84          	lea    (%r12,%rax,4),%rdx
    6cc5:	4c 89 c8             	mov    %r9,%rax
    6cc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    6ccf:	00 
    6cd0:	8b 0a                	mov    (%rdx),%ecx
    6cd2:	48 83 c2 04          	add    $0x4,%rdx
    6cd6:	31 08                	xor    %ecx,(%rax)
              for (z = 0; z < k*w; z++) {
    6cd8:	48 83 c0 04          	add    $0x4,%rax
    6cdc:	48 39 c6             	cmp    %rax,%rsi
    6cdf:	75 ef                	jne    6cd0 <jerasure_generate_decoding_schedule+0x440>
    6ce1:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
          for (y = 0; y < w; y++) {
    6ce6:	01 ef                	add    %ebp,%edi
    6ce8:	48 83 c1 04          	add    $0x4,%rcx
    6cec:	49 39 c8             	cmp    %rcx,%r8
    6cef:	75 ba                	jne    6cab <jerasure_generate_decoding_schedule+0x41b>
        for (j = 0; j < w; j++) {
    6cf1:	41 ff c3             	inc    %r11d
    6cf4:	4d 01 e8             	add    %r13,%r8
    6cf7:	4d 01 e9             	add    %r13,%r9
    6cfa:	4c 01 ee             	add    %r13,%rsi
    6cfd:	44 39 db             	cmp    %r11d,%ebx
    6d00:	75 8e                	jne    6c90 <jerasure_generate_decoding_schedule+0x400>
    6d02:	4c 8b 7c 24 38       	mov    0x38(%rsp),%r15
    for (i = 0; i < k; i++) {
    6d07:	01 5c 24 28          	add    %ebx,0x28(%rsp)
    6d0b:	49 8d 47 01          	lea    0x1(%r15),%rax
    6d0f:	4c 39 7c 24 18       	cmp    %r15,0x18(%rsp)
    6d14:	0f 85 0a ff ff ff    	jne    6c24 <jerasure_generate_decoding_schedule+0x394>
  for (x = 0; x < cdf; x++) {
    6d1a:	48 83 44 24 70 04    	addq   $0x4,0x70(%rsp)
    6d20:	48 8b b4 24 a0 00 00 	mov    0xa0(%rsp),%rsi
    6d27:	00 
    6d28:	48 01 74 24 30       	add    %rsi,0x30(%rsp)
    6d2d:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
    6d32:	48 01 74 24 50       	add    %rsi,0x50(%rsp)
    6d37:	48 39 84 24 98 00 00 	cmp    %rax,0x98(%rsp)
    6d3e:	00 
    6d3f:	0f 85 cd fd ff ff    	jne    6b12 <jerasure_generate_decoding_schedule+0x282>
  if (smart) {
    6d45:	8b 94 24 ac 00 00 00 	mov    0xac(%rsp),%edx
    6d4c:	85 d2                	test   %edx,%edx
    6d4e:	75 4d                	jne    6d9d <jerasure_generate_decoding_schedule+0x50d>
    schedule = jerasure_dumb_bitmatrix_to_schedule(k, ddf+cdf, w, real_decoding_matrix);
    6d50:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    6d55:	8b b4 24 a8 00 00 00 	mov    0xa8(%rsp),%esi
    6d5c:	8b 7c 24 24          	mov    0x24(%rsp),%edi
    6d60:	89 da                	mov    %ebx,%edx
    6d62:	e8 59 f4 ff ff       	callq  61c0 <jerasure_dumb_bitmatrix_to_schedule>
    6d67:	49 89 c4             	mov    %rax,%r12
  free(row_ids);
    6d6a:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    6d6f:	e8 0c a5 ff ff       	callq  1280 <free@plt>
  free(ind_to_row);
    6d74:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    6d79:	e8 02 a5 ff ff       	callq  1280 <free@plt>
  free(real_decoding_matrix);
    6d7e:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
    6d83:	e8 f8 a4 ff ff       	callq  1280 <free@plt>
}
    6d88:	48 81 c4 b8 00 00 00 	add    $0xb8,%rsp
    6d8f:	5b                   	pop    %rbx
    6d90:	5d                   	pop    %rbp
    6d91:	4c 89 e0             	mov    %r12,%rax
    6d94:	41 5c                	pop    %r12
    6d96:	41 5d                	pop    %r13
    6d98:	41 5e                	pop    %r14
    6d9a:	41 5f                	pop    %r15
    6d9c:	c3                   	retq   
    schedule = jerasure_smart_bitmatrix_to_schedule(k, ddf+cdf, w, real_decoding_matrix);
    6d9d:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    6da2:	8b b4 24 a8 00 00 00 	mov    0xa8(%rsp),%esi
    6da9:	8b 7c 24 24          	mov    0x24(%rsp),%edi
    6dad:	89 da                	mov    %ebx,%edx
    6daf:	e8 9c f5 ff ff       	callq  6350 <jerasure_smart_bitmatrix_to_schedule>
    6db4:	49 89 c4             	mov    %rax,%r12
    6db7:	eb b1                	jmp    6d6a <jerasure_generate_decoding_schedule+0x4da>
    decoding_matrix = talloc(int, k*k*w*w);
    6db9:	44 8b 7c 24 24       	mov    0x24(%rsp),%r15d
    6dbe:	44 89 f8             	mov    %r15d,%eax
    6dc1:	0f af c3             	imul   %ebx,%eax
    6dc4:	0f af c0             	imul   %eax,%eax
    6dc7:	4c 63 e8             	movslq %eax,%r13
    6dca:	49 c1 e5 02          	shl    $0x2,%r13
    6dce:	4c 89 ef             	mov    %r13,%rdi
    6dd1:	e8 2a a6 ff ff       	callq  1400 <malloc@plt>
    6dd6:	49 89 c4             	mov    %rax,%r12
    for (i = 0; i < k; i++) {
    6dd9:	45 85 ff             	test   %r15d,%r15d
    6ddc:	0f 8e b8 00 00 00    	jle    6e9a <jerasure_generate_decoding_schedule+0x60a>
        memcpy(ptr, bitmatrix+k*w*w*(row_ids[i]-k), k*w*w*sizeof(int));
    6de2:	44 89 f0             	mov    %r14d,%eax
    6de5:	0f af c3             	imul   %ebx,%eax
    6de8:	89 6c 24 50          	mov    %ebp,0x50(%rsp)
    6dec:	c7 44 24 18 00 00 00 	movl   $0x0,0x18(%rsp)
    6df3:	00 
    6df4:	48 63 d0             	movslq %eax,%rdx
    6df7:	89 44 24 28          	mov    %eax,0x28(%rsp)
    6dfb:	8b 44 24 24          	mov    0x24(%rsp),%eax
    6dff:	48 c1 e2 02          	shl    $0x2,%rdx
    6e03:	ff c8                	dec    %eax
    6e05:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    6e0a:	49 63 c6             	movslq %r14d,%rax
    6e0d:	48 8d 04 85 04 00 00 	lea    0x4(,%rax,4),%rax
    6e14:	00 
    6e15:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    ptr = decoding_matrix;
    6e1a:	4c 89 e7             	mov    %r12,%rdi
        memcpy(ptr, bitmatrix+k*w*w*(row_ids[i]-k), k*w*w*sizeof(int));
    6e1d:	45 31 ff             	xor    %r15d,%r15d
    6e20:	48 89 d5             	mov    %rdx,%rbp
    6e23:	eb 34                	jmp    6e59 <jerasure_generate_decoding_schedule+0x5c9>
    6e25:	2b 44 24 24          	sub    0x24(%rsp),%eax
    6e29:	0f af 44 24 28       	imul   0x28(%rsp),%eax
    6e2e:	48 8b 74 24 78       	mov    0x78(%rsp),%rsi
    6e33:	48 89 ea             	mov    %rbp,%rdx
    6e36:	48 98                	cltq   
    6e38:	48 8d 34 86          	lea    (%rsi,%rax,4),%rsi
    6e3c:	e8 7f a5 ff ff       	callq  13c0 <memcpy@plt>
    6e41:	48 89 c7             	mov    %rax,%rdi
      ptr += (k*w*w);
    6e44:	01 5c 24 18          	add    %ebx,0x18(%rsp)
    6e48:	48 01 ef             	add    %rbp,%rdi
    for (i = 0; i < k; i++) {
    6e4b:	49 8d 47 01          	lea    0x1(%r15),%rax
    6e4f:	4c 3b 7c 24 38       	cmp    0x38(%rsp),%r15
    6e54:	74 40                	je     6e96 <jerasure_generate_decoding_schedule+0x606>
    6e56:	49 89 c7             	mov    %rax,%r15
      if (row_ids[i] == i) {
    6e59:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    6e5e:	42 8b 04 b8          	mov    (%rax,%r15,4),%eax
    6e62:	44 39 f8             	cmp    %r15d,%eax
    6e65:	75 be                	jne    6e25 <jerasure_generate_decoding_schedule+0x595>
    6e67:	48 89 ea             	mov    %rbp,%rdx
    6e6a:	31 f6                	xor    %esi,%esi
    6e6c:	e8 df a4 ff ff       	callq  1350 <memset@plt>
    6e71:	48 89 c7             	mov    %rax,%rdi
        for (x = 0; x < w; x++) {
    6e74:	85 db                	test   %ebx,%ebx
    6e76:	7e cc                	jle    6e44 <jerasure_generate_decoding_schedule+0x5b4>
    6e78:	48 63 44 24 18       	movslq 0x18(%rsp),%rax
    6e7d:	48 8d 0c 87          	lea    (%rdi,%rax,4),%rcx
    6e81:	31 c0                	xor    %eax,%eax
    6e83:	ff c0                	inc    %eax
          ptr[x+i*w+x*k*w] = 1;
    6e85:	c7 01 01 00 00 00    	movl   $0x1,(%rcx)
        for (x = 0; x < w; x++) {
    6e8b:	48 03 4c 24 30       	add    0x30(%rsp),%rcx
    6e90:	39 c3                	cmp    %eax,%ebx
    6e92:	75 ef                	jne    6e83 <jerasure_generate_decoding_schedule+0x5f3>
    6e94:	eb ae                	jmp    6e44 <jerasure_generate_decoding_schedule+0x5b4>
    6e96:	8b 6c 24 50          	mov    0x50(%rsp),%ebp
    inverse = talloc(int, k*k*w*w);
    6e9a:	4c 89 ef             	mov    %r13,%rdi
    6e9d:	e8 5e a5 ff ff       	callq  1400 <malloc@plt>
    jerasure_invert_bitmatrix(decoding_matrix, inverse, k*w);
    6ea2:	48 89 c6             	mov    %rax,%rsi
    6ea5:	44 89 f2             	mov    %r14d,%edx
    6ea8:	4c 89 e7             	mov    %r12,%rdi
    6eab:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    6eb0:	e8 0b e1 ff ff       	callq  4fc0 <jerasure_invert_bitmatrix>
    free(decoding_matrix);
    6eb5:	4c 89 e7             	mov    %r12,%rdi
    6eb8:	e8 c3 a3 ff ff       	callq  1280 <free@plt>
      memcpy(ptr, inverse+k*w*w*row_ids[k+i], sizeof(int)*k*w*w);
    6ebd:	48 63 c3             	movslq %ebx,%rax
    6ec0:	48 0f af c0          	imul   %rax,%rax
    6ec4:	48 63 74 24 24       	movslq 0x24(%rsp),%rsi
    6ec9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    6ece:	48 0f af c6          	imul   %rsi,%rax
    6ed2:	45 89 f4             	mov    %r14d,%r12d
    6ed5:	4c 8d 3c b7          	lea    (%rdi,%rsi,4),%r15
    6ed9:	48 89 c2             	mov    %rax,%rdx
    6edc:	8d 45 ff             	lea    -0x1(%rbp),%eax
    6edf:	44 0f af e3          	imul   %ebx,%r12d
    6ee3:	48 01 c6             	add    %rax,%rsi
    6ee6:	48 8d 44 b7 04       	lea    0x4(%rdi,%rsi,4),%rax
    6eeb:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
      ptr += (k*w*w);
    6ef0:	4d 63 ec             	movslq %r12d,%r13
    ptr = real_decoding_matrix;
    6ef3:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
    6ef8:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
      memcpy(ptr, inverse+k*w*w*row_ids[k+i], sizeof(int)*k*w*w);
    6efd:	48 c1 e2 02          	shl    $0x2,%rdx
      ptr += (k*w*w);
    6f01:	49 c1 e5 02          	shl    $0x2,%r13
      memcpy(ptr, inverse+k*w*w*row_ids[k+i], sizeof(int)*k*w*w);
    6f05:	41 8b 07             	mov    (%r15),%eax
    6f08:	4c 89 44 24 30       	mov    %r8,0x30(%rsp)
    6f0d:	41 0f af c4          	imul   %r12d,%eax
    6f11:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
    6f16:	49 83 c7 04          	add    $0x4,%r15
    6f1a:	48 98                	cltq   
    6f1c:	49 8d 34 80          	lea    (%r8,%rax,4),%rsi
    6f20:	e8 9b a4 ff ff       	callq  13c0 <memcpy@plt>
    6f25:	48 89 c7             	mov    %rax,%rdi
      ptr += (k*w*w);
    6f28:	4c 01 ef             	add    %r13,%rdi
    for (i = 0; i < ddf; i++) {
    6f2b:	4c 39 7c 24 18       	cmp    %r15,0x18(%rsp)
    6f30:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    6f35:	4c 8b 44 24 30       	mov    0x30(%rsp),%r8
    6f3a:	75 c9                	jne    6f05 <jerasure_generate_decoding_schedule+0x675>
    free(inverse);
    6f3c:	4c 89 c7             	mov    %r8,%rdi
    6f3f:	e8 3c a3 ff ff       	callq  1280 <free@plt>
    6f44:	e9 f7 fa ff ff       	jmpq   6a40 <jerasure_generate_decoding_schedule+0x1b0>
  for (i = 0; i < k; i++) {
    6f49:	8b 74 24 24          	mov    0x24(%rsp),%esi
    6f4d:	e9 7f fa ff ff       	jmpq   69d1 <jerasure_generate_decoding_schedule+0x141>
  cdf = 0;
    6f52:	c7 44 24 08 00 00 00 	movl   $0x0,0x8(%rsp)
    6f59:	00 
  ddf = 0;
    6f5a:	31 ed                	xor    %ebp,%ebp
    6f5c:	e9 97 f9 ff ff       	jmpq   68f8 <jerasure_generate_decoding_schedule+0x68>
    6f61:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    6f68:	00 00 00 00 
    6f6c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000006f70 <jerasure_schedule_decode_lazy>:
{
    6f70:	f3 0f 1e fa          	endbr64 
    6f74:	41 57                	push   %r15
    6f76:	4d 89 c7             	mov    %r8,%r15
    6f79:	41 56                	push   %r14
    6f7b:	41 55                	push   %r13
    6f7d:	41 54                	push   %r12
    6f7f:	49 89 cc             	mov    %rcx,%r12
    6f82:	4c 89 c9             	mov    %r9,%rcx
    6f85:	55                   	push   %rbp
    6f86:	89 fd                	mov    %edi,%ebp
    6f88:	53                   	push   %rbx
    6f89:	89 f3                	mov    %esi,%ebx
    6f8b:	48 83 ec 18          	sub    $0x18,%rsp
  ptrs = set_up_ptrs_for_scheduled_decoding(k, m, erasures, data_ptrs, coding_ptrs);
    6f8f:	4c 8b 44 24 50       	mov    0x50(%rsp),%r8
{
    6f94:	89 54 24 08          	mov    %edx,0x8(%rsp)
  ptrs = set_up_ptrs_for_scheduled_decoding(k, m, erasures, data_ptrs, coding_ptrs);
    6f98:	4c 89 fa             	mov    %r15,%rdx
{
    6f9b:	44 8b 74 24 60       	mov    0x60(%rsp),%r14d
  ptrs = set_up_ptrs_for_scheduled_decoding(k, m, erasures, data_ptrs, coding_ptrs);
    6fa0:	e8 bb d5 ff ff       	callq  4560 <set_up_ptrs_for_scheduled_decoding>
  if (ptrs == NULL) return -1;
    6fa5:	48 85 c0             	test   %rax,%rax
    6fa8:	0f 84 a9 00 00 00    	je     7057 <jerasure_schedule_decode_lazy+0xe7>
  schedule = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    6fae:	44 8b 4c 24 68       	mov    0x68(%rsp),%r9d
    6fb3:	8b 54 24 08          	mov    0x8(%rsp),%edx
    6fb7:	4c 89 e1             	mov    %r12,%rcx
    6fba:	4d 89 f8             	mov    %r15,%r8
    6fbd:	89 de                	mov    %ebx,%esi
    6fbf:	89 ef                	mov    %ebp,%edi
    6fc1:	49 89 c5             	mov    %rax,%r13
    6fc4:	e8 c7 f8 ff ff       	callq  6890 <jerasure_generate_decoding_schedule>
    6fc9:	49 89 c4             	mov    %rax,%r12
  if (schedule == NULL) {
    6fcc:	48 85 c0             	test   %rax,%rax
    6fcf:	0f 84 89 00 00 00    	je     705e <jerasure_schedule_decode_lazy+0xee>
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    6fd5:	8b 54 24 58          	mov    0x58(%rsp),%edx
    6fd9:	85 d2                	test   %edx,%edx
    6fdb:	7e 59                	jle    7036 <jerasure_schedule_decode_lazy+0xc6>
    for (i = 0; i < k+m; i++) ptrs[i] += (packetsize*w);
    6fdd:	8b 44 24 08          	mov    0x8(%rsp),%eax
    6fe1:	41 0f af c6          	imul   %r14d,%eax
    6fe5:	89 44 24 08          	mov    %eax,0x8(%rsp)
    6fe9:	4c 63 f8             	movslq %eax,%r15
    6fec:	8d 44 1d 00          	lea    0x0(%rbp,%rbx,1),%eax
    6ff0:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    6ff4:	ff c8                	dec    %eax
    6ff6:	49 8d 5c c5 08       	lea    0x8(%r13,%rax,8),%rbx
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    6ffb:	31 ed                	xor    %ebp,%ebp
    6ffd:	0f 1f 00             	nopl   (%rax)
  jerasure_do_scheduled_operations(ptrs, schedule, packetsize);
    7000:	44 89 f2             	mov    %r14d,%edx
    7003:	4c 89 e6             	mov    %r12,%rsi
    7006:	4c 89 ef             	mov    %r13,%rdi
    7009:	e8 b2 ec ff ff       	callq  5cc0 <jerasure_do_scheduled_operations>
    for (i = 0; i < k+m; i++) ptrs[i] += (packetsize*w);
    700e:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    7012:	4c 89 ea             	mov    %r13,%rdx
    7015:	85 c0                	test   %eax,%eax
    7017:	7e 13                	jle    702c <jerasure_schedule_decode_lazy+0xbc>
    7019:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7020:	4c 01 3a             	add    %r15,(%rdx)
    7023:	48 83 c2 08          	add    $0x8,%rdx
    7027:	48 39 da             	cmp    %rbx,%rdx
    702a:	75 f4                	jne    7020 <jerasure_schedule_decode_lazy+0xb0>
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    702c:	03 6c 24 08          	add    0x8(%rsp),%ebp
    7030:	39 6c 24 58          	cmp    %ebp,0x58(%rsp)
    7034:	7f ca                	jg     7000 <jerasure_schedule_decode_lazy+0x90>
  jerasure_free_schedule(schedule);
    7036:	4c 89 e7             	mov    %r12,%rdi
    7039:	e8 52 d6 ff ff       	callq  4690 <jerasure_free_schedule>
  free(ptrs);
    703e:	4c 89 ef             	mov    %r13,%rdi
    7041:	e8 3a a2 ff ff       	callq  1280 <free@plt>
  return 0;
    7046:	31 c0                	xor    %eax,%eax
}
    7048:	48 83 c4 18          	add    $0x18,%rsp
    704c:	5b                   	pop    %rbx
    704d:	5d                   	pop    %rbp
    704e:	41 5c                	pop    %r12
    7050:	41 5d                	pop    %r13
    7052:	41 5e                	pop    %r14
    7054:	41 5f                	pop    %r15
    7056:	c3                   	retq   
  if (ptrs == NULL) return -1;
    7057:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    705c:	eb ea                	jmp    7048 <jerasure_schedule_decode_lazy+0xd8>
    free(ptrs);
    705e:	4c 89 ef             	mov    %r13,%rdi
    7061:	e8 1a a2 ff ff       	callq  1280 <free@plt>
    return -1;
    7066:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    706b:	eb db                	jmp    7048 <jerasure_schedule_decode_lazy+0xd8>
    706d:	0f 1f 00             	nopl   (%rax)

0000000000007070 <jerasure_generate_schedule_cache>:
{
    7070:	f3 0f 1e fa          	endbr64 
    7074:	41 57                	push   %r15
    7076:	41 56                	push   %r14
    7078:	41 55                	push   %r13
    707a:	41 54                	push   %r12
    707c:	55                   	push   %rbp
    707d:	53                   	push   %rbx
    707e:	48 83 ec 78          	sub    $0x78,%rsp
    7082:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    7089:	00 00 
    708b:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
    7090:	31 c0                	xor    %eax,%eax
  if (m != 2) return NULL;
    7092:	83 fe 02             	cmp    $0x2,%esi
    7095:	0f 85 65 01 00 00    	jne    7200 <jerasure_generate_schedule_cache+0x190>
  scache = talloc(int **, (k+m)*(k+m+1));
    709b:	8d 5f 02             	lea    0x2(%rdi),%ebx
    709e:	89 fd                	mov    %edi,%ebp
    70a0:	8d 7f 03             	lea    0x3(%rdi),%edi
    70a3:	0f af fb             	imul   %ebx,%edi
    70a6:	41 89 d6             	mov    %edx,%r14d
    70a9:	49 89 cf             	mov    %rcx,%r15
    70ac:	48 63 ff             	movslq %edi,%rdi
    70af:	48 c1 e7 03          	shl    $0x3,%rdi
    70b3:	45 89 c5             	mov    %r8d,%r13d
    70b6:	e8 45 a3 ff ff       	callq  1400 <malloc@plt>
    70bb:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  if (scache == NULL) return NULL;
    70c0:	48 85 c0             	test   %rax,%rax
    70c3:	0f 84 37 01 00 00    	je     7200 <jerasure_generate_schedule_cache+0x190>
  for (e1 = 0; e1 < k+m; e1++) {
    70c9:	85 db                	test   %ebx,%ebx
    70cb:	0f 8e 38 01 00 00    	jle    7209 <jerasure_generate_schedule_cache+0x199>
    erasures[0] = e1;
    70d1:	48 63 db             	movslq %ebx,%rbx
    70d4:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    70d9:	48 8d 04 dd 08 00 00 	lea    0x8(,%rbx,8),%rax
    70e0:	00 
    70e1:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    70e6:	48 8d 04 dd 00 00 00 	lea    0x0(,%rbx,8),%rax
    70ed:	00 
    70ee:	48 89 c6             	mov    %rax,%rsi
    70f1:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    70f6:	48 89 c8             	mov    %rcx,%rax
    70f9:	48 01 f0             	add    %rsi,%rax
    70fc:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    7101:	8d 45 01             	lea    0x1(%rbp),%eax
    7104:	48 ff c0             	inc    %rax
    7107:	c7 44 24 5c 00 00 00 	movl   $0x0,0x5c(%rsp)
    710e:	00 
    for (e2 = 0; e2 < e1; e2++) {
    710f:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    erasures[0] = e1;
    7114:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
    7119:	48 c7 44 24 10 01 00 	movq   $0x1,0x10(%rsp)
    7120:	00 00 
    7122:	89 6c 24 34          	mov    %ebp,0x34(%rsp)
    7126:	4c 8d 44 24 5c       	lea    0x5c(%rsp),%r8
    712b:	4d 89 c4             	mov    %r8,%r12
    712e:	66 90                	xchg   %ax,%ax
    scache[e1*(k+m)+e1] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    7130:	8b 7c 24 34          	mov    0x34(%rsp),%edi
    7134:	4c 89 f9             	mov    %r15,%rcx
    7137:	be 02 00 00 00       	mov    $0x2,%esi
    713c:	45 89 e9             	mov    %r13d,%r9d
    713f:	4d 89 e0             	mov    %r12,%r8
    7142:	44 89 f2             	mov    %r14d,%edx
    erasures[1] = -1;
    7145:	c7 44 24 60 ff ff ff 	movl   $0xffffffff,0x60(%rsp)
    714c:	ff 
    scache[e1*(k+m)+e1] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    714d:	e8 3e f7 ff ff       	callq  6890 <jerasure_generate_decoding_schedule>
    7152:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  for (e1 = 0; e1 < k+m; e1++) {
    7157:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
    scache[e1*(k+m)+e1] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    715c:	48 89 01             	mov    %rax,(%rcx)
  for (e1 = 0; e1 < k+m; e1++) {
    715f:	48 39 74 24 10       	cmp    %rsi,0x10(%rsp)
    7164:	0f 84 9f 00 00 00    	je     7209 <jerasure_generate_schedule_cache+0x199>
    erasures[0] = e1;
    716a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    716f:	89 44 24 5c          	mov    %eax,0x5c(%rsp)
    for (e2 = 0; e2 < e1; e2++) {
    7173:	85 c0                	test   %eax,%eax
    7175:	7e 64                	jle    71db <jerasure_generate_schedule_cache+0x16b>
    7177:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    717c:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    7181:	4d 89 e0             	mov    %r12,%r8
    7184:	44 8b 64 24 34       	mov    0x34(%rsp),%r12d
    7189:	48 8d 2c f8          	lea    (%rax,%rdi,8),%rbp
    718d:	31 db                	xor    %ebx,%ebx
    718f:	90                   	nop
      scache[e1*(k+m)+e2] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    7190:	44 89 f2             	mov    %r14d,%edx
    7193:	45 89 e9             	mov    %r13d,%r9d
    7196:	4c 89 f9             	mov    %r15,%rcx
    7199:	be 02 00 00 00       	mov    $0x2,%esi
    719e:	44 89 e7             	mov    %r12d,%edi
      erasures[1] = e2;
    71a1:	89 5c 24 60          	mov    %ebx,0x60(%rsp)
      scache[e1*(k+m)+e2] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    71a5:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
      erasures[2] = -1;
    71aa:	c7 44 24 64 ff ff ff 	movl   $0xffffffff,0x64(%rsp)
    71b1:	ff 
      scache[e1*(k+m)+e2] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    71b2:	e8 d9 f6 ff ff       	callq  6890 <jerasure_generate_decoding_schedule>
    71b7:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
    for (e2 = 0; e2 < e1; e2++) {
    71bc:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
      scache[e1*(k+m)+e2] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    71c1:	48 89 04 da          	mov    %rax,(%rdx,%rbx,8)
      scache[e2*(k+m)+e1] = scache[e1*(k+m)+e2];
    71c5:	48 ff c3             	inc    %rbx
    71c8:	48 89 45 00          	mov    %rax,0x0(%rbp)
    for (e2 = 0; e2 < e1; e2++) {
    71cc:	48 03 6c 24 20       	add    0x20(%rsp),%rbp
    71d1:	48 3b 5c 24 10       	cmp    0x10(%rsp),%rbx
    71d6:	75 b8                	jne    7190 <jerasure_generate_schedule_cache+0x120>
    71d8:	4d 89 c4             	mov    %r8,%r12
    71db:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    71e0:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
    71e5:	48 ff 44 24 10       	incq   0x10(%rsp)
    71ea:	48 01 4c 24 28       	add    %rcx,0x28(%rsp)
    71ef:	48 01 74 24 18       	add    %rsi,0x18(%rsp)
    71f4:	e9 37 ff ff ff       	jmpq   7130 <jerasure_generate_schedule_cache+0xc0>
    71f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  if (m != 2) return NULL;
    7200:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
    7207:	00 00 
}
    7209:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
    720e:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    7215:	00 00 
    7217:	75 14                	jne    722d <jerasure_generate_schedule_cache+0x1bd>
    7219:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    721e:	48 83 c4 78          	add    $0x78,%rsp
    7222:	5b                   	pop    %rbx
    7223:	5d                   	pop    %rbp
    7224:	41 5c                	pop    %r12
    7226:	41 5d                	pop    %r13
    7228:	41 5e                	pop    %r14
    722a:	41 5f                	pop    %r15
    722c:	c3                   	retq   
    722d:	e8 de a0 ff ff       	callq  1310 <__stack_chk_fail@plt>
    7232:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    7239:	00 00 00 00 
    723d:	0f 1f 00             	nopl   (%rax)

0000000000007240 <jerasure_bitmatrix_encode>:

void jerasure_bitmatrix_encode(int k, int m, int w, int *bitmatrix,
                            char **data_ptrs, char **coding_ptrs, int size, int packetsize)
{
    7240:	f3 0f 1e fa          	endbr64 
    7244:	41 57                	push   %r15
    7246:	41 56                	push   %r14
    7248:	41 55                	push   %r13
    724a:	41 54                	push   %r12
    724c:	55                   	push   %rbp
    724d:	53                   	push   %rbx
    724e:	48 83 ec 18          	sub    $0x18,%rsp
  int i, j, x, y, sptr, pstarted, index;
  char *dptr, *pptr;

  if (packetsize%sizeof(long) != 0) {
    7252:	f6 44 24 58 07       	testb  $0x7,0x58(%rsp)
    7257:	0f 85 9a 00 00 00    	jne    72f7 <jerasure_bitmatrix_encode+0xb7>
    725d:	4d 89 c6             	mov    %r8,%r14
    fprintf(stderr, "jerasure_bitmatrix_encode - packetsize(%d) %c sizeof(long) != 0\n", packetsize, '%');
    exit(1);
  }
  if (size%(packetsize*w) != 0) {
    7260:	44 8b 44 24 58       	mov    0x58(%rsp),%r8d
    7265:	8b 44 24 50          	mov    0x50(%rsp),%eax
    7269:	44 0f af c2          	imul   %edx,%r8d
    726d:	41 89 d4             	mov    %edx,%r12d
    7270:	99                   	cltd   
    7271:	41 f7 f8             	idiv   %r8d
    7274:	85 d2                	test   %edx,%edx
    7276:	0f 85 a9 00 00 00    	jne    7325 <jerasure_bitmatrix_encode+0xe5>
    fprintf(stderr, "jerasure_bitmatrix_encode - size(%d) %c (packetsize(%d)*w(%d))) != 0\n", 
         size, '%', packetsize, w);
    exit(1);
  }

  for (i = 0; i < m; i++) {
    727c:	85 f6                	test   %esi,%esi
    727e:	7e 68                	jle    72e8 <jerasure_bitmatrix_encode+0xa8>
    7280:	44 89 e3             	mov    %r12d,%ebx
    7283:	0f af df             	imul   %edi,%ebx
    7286:	89 fd                	mov    %edi,%ebp
    7288:	4d 89 cf             	mov    %r9,%r15
    728b:	41 0f af dc          	imul   %r12d,%ebx
    728f:	41 89 fd             	mov    %edi,%r13d
    7292:	48 63 db             	movslq %ebx,%rbx
    7295:	48 8d 04 9d 00 00 00 	lea    0x0(,%rbx,4),%rax
    729c:	00 
    729d:	48 89 04 24          	mov    %rax,(%rsp)
    72a1:	8d 04 3e             	lea    (%rsi,%rdi,1),%eax
    72a4:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    72a8:	48 89 cb             	mov    %rcx,%rbx
    72ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    jerasure_bitmatrix_dotprod(k, w, bitmatrix+i*k*w*w, NULL, k+i, data_ptrs, coding_ptrs, size, packetsize);
    72b0:	48 83 ec 08          	sub    $0x8,%rsp
    72b4:	8b 44 24 60          	mov    0x60(%rsp),%eax
    72b8:	45 89 e8             	mov    %r13d,%r8d
    72bb:	50                   	push   %rax
    72bc:	48 89 da             	mov    %rbx,%rdx
    72bf:	4d 89 f1             	mov    %r14,%r9
    72c2:	8b 44 24 60          	mov    0x60(%rsp),%eax
    72c6:	31 c9                	xor    %ecx,%ecx
    72c8:	50                   	push   %rax
    72c9:	44 89 e6             	mov    %r12d,%esi
    72cc:	89 ef                	mov    %ebp,%edi
    72ce:	41 57                	push   %r15
    72d0:	41 ff c5             	inc    %r13d
    72d3:	e8 e8 c4 ff ff       	callq  37c0 <jerasure_bitmatrix_dotprod>
  for (i = 0; i < m; i++) {
    72d8:	48 03 5c 24 20       	add    0x20(%rsp),%rbx
    72dd:	48 83 c4 20          	add    $0x20,%rsp
    72e1:	44 3b 6c 24 0c       	cmp    0xc(%rsp),%r13d
    72e6:	75 c8                	jne    72b0 <jerasure_bitmatrix_encode+0x70>
  }
}
    72e8:	48 83 c4 18          	add    $0x18,%rsp
    72ec:	5b                   	pop    %rbx
    72ed:	5d                   	pop    %rbp
    72ee:	41 5c                	pop    %r12
    72f0:	41 5d                	pop    %r13
    72f2:	41 5e                	pop    %r14
    72f4:	41 5f                	pop    %r15
    72f6:	c3                   	retq   
    72f7:	48 8b 3d 42 9e 00 00 	mov    0x9e42(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    72fe:	8b 4c 24 58          	mov    0x58(%rsp),%ecx
    7302:	41 b8 25 00 00 00    	mov    $0x25,%r8d
    7308:	48 8d 15 e1 34 00 00 	lea    0x34e1(%rip),%rdx        # a7f0 <__PRETTY_FUNCTION__.5741+0x117>
    730f:	be 01 00 00 00       	mov    $0x1,%esi
    7314:	31 c0                	xor    %eax,%eax
    7316:	e8 65 a1 ff ff       	callq  1480 <__fprintf_chk@plt>
    exit(1);
    731b:	bf 01 00 00 00       	mov    $0x1,%edi
    7320:	e8 3b a1 ff ff       	callq  1460 <exit@plt>
    7325:	50                   	push   %rax
    7326:	48 8b 3d 13 9e 00 00 	mov    0x9e13(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    732d:	41 b8 25 00 00 00    	mov    $0x25,%r8d
    7333:	41 54                	push   %r12
    7335:	48 8d 15 fc 34 00 00 	lea    0x34fc(%rip),%rdx        # a838 <__PRETTY_FUNCTION__.5741+0x15f>
    733c:	be 01 00 00 00       	mov    $0x1,%esi
    7341:	44 8b 4c 24 68       	mov    0x68(%rsp),%r9d
    7346:	8b 4c 24 60          	mov    0x60(%rsp),%ecx
    734a:	31 c0                	xor    %eax,%eax
    734c:	e8 2f a1 ff ff       	callq  1480 <__fprintf_chk@plt>
    exit(1);
    7351:	bf 01 00 00 00       	mov    $0x1,%edi
    7356:	e8 05 a1 ff ff       	callq  1460 <exit@plt>
    735b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000007360 <galois_create_log_tables.part.0>:
    7360:	41 56                	push   %r14
    7362:	48 8d 05 57 38 00 00 	lea    0x3857(%rip),%rax        # abc0 <nw>
    7369:	4c 8d 35 b0 a1 00 00 	lea    0xa1b0(%rip),%r14        # 11520 <galois_log_tables>
    7370:	41 55                	push   %r13
    7372:	41 54                	push   %r12
    7374:	4c 63 e7             	movslq %edi,%r12
    7377:	4e 63 2c a0          	movslq (%rax,%r12,4),%r13
    737b:	55                   	push   %rbp
    737c:	4a 8d 3c ad 00 00 00 	lea    0x0(,%r13,4),%rdi
    7383:	00 
    7384:	53                   	push   %rbx
    7385:	e8 76 a0 ff ff       	callq  1400 <malloc@plt>
    738a:	4b 89 04 e6          	mov    %rax,(%r14,%r12,8)
    738e:	48 85 c0             	test   %rax,%rax
    7391:	0f 84 31 01 00 00    	je     74c8 <galois_create_log_tables.part.0+0x168>
    7397:	4b 8d 7c 6d 00       	lea    0x0(%r13,%r13,2),%rdi
    739c:	48 c1 e7 02          	shl    $0x2,%rdi
    73a0:	4c 89 eb             	mov    %r13,%rbx
    73a3:	48 89 c5             	mov    %rax,%rbp
    73a6:	4c 8d 2d 53 a0 00 00 	lea    0xa053(%rip),%r13        # 11400 <galois_ilog_tables>
    73ad:	e8 4e a0 ff ff       	callq  1400 <malloc@plt>
    73b2:	4b 89 44 e5 00       	mov    %rax,0x0(%r13,%r12,8)
    73b7:	48 85 c0             	test   %rax,%rax
    73ba:	0f 84 f3 00 00 00    	je     74b3 <galois_create_log_tables.part.0+0x153>
    73c0:	48 8d 15 59 37 00 00 	lea    0x3759(%rip),%rdx        # ab20 <nwm1>
    73c7:	42 8b 34 a2          	mov    (%rdx,%r12,4),%esi
    73cb:	8d 7b ff             	lea    -0x1(%rbx),%edi
    73ce:	31 d2                	xor    %edx,%edx
    73d0:	85 db                	test   %ebx,%ebx
    73d2:	7e 1a                	jle    73ee <galois_create_log_tables.part.0+0x8e>
    73d4:	0f 1f 40 00          	nopl   0x0(%rax)
    73d8:	48 89 d1             	mov    %rdx,%rcx
    73db:	89 74 95 00          	mov    %esi,0x0(%rbp,%rdx,4)
    73df:	c7 04 90 00 00 00 00 	movl   $0x0,(%rax,%rdx,4)
    73e6:	48 ff c2             	inc    %rdx
    73e9:	48 39 cf             	cmp    %rcx,%rdi
    73ec:	75 ea                	jne    73d8 <galois_create_log_tables.part.0+0x78>
    73ee:	48 63 ce             	movslq %esi,%rcx
    73f1:	85 f6                	test   %esi,%esi
    73f3:	7e 76                	jle    746b <galois_create_log_tables.part.0+0x10b>
    73f5:	48 89 c2             	mov    %rax,%rdx
    73f8:	48 89 c7             	mov    %rax,%rdi
    73fb:	31 c9                	xor    %ecx,%ecx
    73fd:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    7403:	4c 8d 35 f6 38 00 00 	lea    0x38f6(%rip),%r14        # ad00 <prim_poly>
    740a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7410:	4d 63 c8             	movslq %r8d,%r9
    7413:	4e 8d 54 8d 00       	lea    0x0(%rbp,%r9,4),%r10
    7418:	45 8b 0a             	mov    (%r10),%r9d
    741b:	47 8d 1c 00          	lea    (%r8,%r8,1),%r11d
    741f:	41 39 f1             	cmp    %esi,%r9d
    7422:	75 5b                	jne    747f <galois_create_log_tables.part.0+0x11f>
    7424:	44 89 07             	mov    %r8d,(%rdi)
    7427:	41 89 0a             	mov    %ecx,(%r10)
    742a:	45 89 d8             	mov    %r11d,%r8d
    742d:	44 85 db             	test   %r11d,%ebx
    7430:	74 07                	je     7439 <galois_create_log_tables.part.0+0xd9>
    7432:	47 33 04 a6          	xor    (%r14,%r12,4),%r8d
    7436:	41 21 f0             	and    %esi,%r8d
    7439:	ff c1                	inc    %ecx
    743b:	48 83 c7 04          	add    $0x4,%rdi
    743f:	39 f1                	cmp    %esi,%ecx
    7441:	75 cd                	jne    7410 <galois_create_log_tables.part.0+0xb0>
    7443:	8d 3c 09             	lea    (%rcx,%rcx,1),%edi
    7446:	8d 71 ff             	lea    -0x1(%rcx),%esi
    7449:	4c 8d 44 b0 04       	lea    0x4(%rax,%rsi,4),%r8
    744e:	48 63 c9             	movslq %ecx,%rcx
    7451:	48 63 ff             	movslq %edi,%rdi
    7454:	0f 1f 40 00          	nopl   0x0(%rax)
    7458:	8b 32                	mov    (%rdx),%esi
    745a:	89 34 8a             	mov    %esi,(%rdx,%rcx,4)
    745d:	8b 32                	mov    (%rdx),%esi
    745f:	89 34 ba             	mov    %esi,(%rdx,%rdi,4)
    7462:	48 83 c2 04          	add    $0x4,%rdx
    7466:	49 39 d0             	cmp    %rdx,%r8
    7469:	75 ed                	jne    7458 <galois_create_log_tables.part.0+0xf8>
    746b:	48 8d 04 88          	lea    (%rax,%rcx,4),%rax
    746f:	4b 89 44 e5 00       	mov    %rax,0x0(%r13,%r12,8)
    7474:	31 c0                	xor    %eax,%eax
    7476:	5b                   	pop    %rbx
    7477:	5d                   	pop    %rbp
    7478:	41 5c                	pop    %r12
    747a:	41 5d                	pop    %r13
    747c:	41 5e                	pop    %r14
    747e:	c3                   	retq   
    747f:	48 8d 05 7a 38 00 00 	lea    0x387a(%rip),%rax        # ad00 <prim_poly>
    7486:	46 33 1c a0          	xor    (%rax,%r12,4),%r11d
    748a:	41 53                	push   %r11
    748c:	48 8d 15 f5 33 00 00 	lea    0x33f5(%rip),%rdx        # a888 <__PRETTY_FUNCTION__.5741+0x1af>
    7493:	be 01 00 00 00       	mov    $0x1,%esi
    7498:	8b 07                	mov    (%rdi),%eax
    749a:	48 8b 3d 9f 9c 00 00 	mov    0x9c9f(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    74a1:	50                   	push   %rax
    74a2:	31 c0                	xor    %eax,%eax
    74a4:	e8 d7 9f ff ff       	callq  1480 <__fprintf_chk@plt>
    74a9:	bf 01 00 00 00       	mov    $0x1,%edi
    74ae:	e8 ad 9f ff ff       	callq  1460 <exit@plt>
    74b3:	48 89 ef             	mov    %rbp,%rdi
    74b6:	e8 c5 9d ff ff       	callq  1280 <free@plt>
    74bb:	4b c7 04 e6 00 00 00 	movq   $0x0,(%r14,%r12,8)
    74c2:	00 
    74c3:	83 c8 ff             	or     $0xffffffff,%eax
    74c6:	eb ae                	jmp    7476 <galois_create_log_tables.part.0+0x116>
    74c8:	83 c8 ff             	or     $0xffffffff,%eax
    74cb:	eb a9                	jmp    7476 <galois_create_log_tables.part.0+0x116>
    74cd:	0f 1f 00             	nopl   (%rax)

00000000000074d0 <galois_create_log_tables>:
    74d0:	f3 0f 1e fa          	endbr64 
    74d4:	83 ff 1e             	cmp    $0x1e,%edi
    74d7:	7f 27                	jg     7500 <galois_create_log_tables+0x30>
    74d9:	48 63 d7             	movslq %edi,%rdx
    74dc:	48 8d 05 3d a0 00 00 	lea    0xa03d(%rip),%rax        # 11520 <galois_log_tables>
    74e3:	45 31 c0             	xor    %r8d,%r8d
    74e6:	48 83 3c d0 00       	cmpq   $0x0,(%rax,%rdx,8)
    74eb:	74 0b                	je     74f8 <galois_create_log_tables+0x28>
    74ed:	44 89 c0             	mov    %r8d,%eax
    74f0:	c3                   	retq   
    74f1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    74f8:	e9 63 fe ff ff       	jmpq   7360 <galois_create_log_tables.part.0>
    74fd:	0f 1f 00             	nopl   (%rax)
    7500:	41 b8 ff ff ff ff    	mov    $0xffffffff,%r8d
    7506:	eb e5                	jmp    74ed <galois_create_log_tables+0x1d>
    7508:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    750f:	00 

0000000000007510 <galois_logtable_multiply>:
    7510:	f3 0f 1e fa          	endbr64 
    7514:	85 ff                	test   %edi,%edi
    7516:	74 38                	je     7550 <galois_logtable_multiply+0x40>
    7518:	85 f6                	test   %esi,%esi
    751a:	74 34                	je     7550 <galois_logtable_multiply+0x40>
    751c:	48 63 d2             	movslq %edx,%rdx
    751f:	48 8d 05 fa 9f 00 00 	lea    0x9ffa(%rip),%rax        # 11520 <galois_log_tables>
    7526:	48 8b 0c d0          	mov    (%rax,%rdx,8),%rcx
    752a:	48 63 f6             	movslq %esi,%rsi
    752d:	8b 04 b1             	mov    (%rcx,%rsi,4),%eax
    7530:	48 63 ff             	movslq %edi,%rdi
    7533:	03 04 b9             	add    (%rcx,%rdi,4),%eax
    7536:	48 8d 0d c3 9e 00 00 	lea    0x9ec3(%rip),%rcx        # 11400 <galois_ilog_tables>
    753d:	48 8b 14 d1          	mov    (%rcx,%rdx,8),%rdx
    7541:	48 98                	cltq   
    7543:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    7546:	c3                   	retq   
    7547:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    754e:	00 00 
    7550:	31 c0                	xor    %eax,%eax
    7552:	c3                   	retq   
    7553:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    755a:	00 00 00 00 
    755e:	66 90                	xchg   %ax,%ax

0000000000007560 <galois_logtable_divide>:
    7560:	f3 0f 1e fa          	endbr64 
    7564:	85 f6                	test   %esi,%esi
    7566:	74 38                	je     75a0 <galois_logtable_divide+0x40>
    7568:	31 c0                	xor    %eax,%eax
    756a:	85 ff                	test   %edi,%edi
    756c:	74 37                	je     75a5 <galois_logtable_divide+0x45>
    756e:	48 63 d2             	movslq %edx,%rdx
    7571:	48 8d 05 a8 9f 00 00 	lea    0x9fa8(%rip),%rax        # 11520 <galois_log_tables>
    7578:	48 8b 0c d0          	mov    (%rax,%rdx,8),%rcx
    757c:	48 63 ff             	movslq %edi,%rdi
    757f:	8b 04 b9             	mov    (%rcx,%rdi,4),%eax
    7582:	48 63 f6             	movslq %esi,%rsi
    7585:	2b 04 b1             	sub    (%rcx,%rsi,4),%eax
    7588:	48 8d 0d 71 9e 00 00 	lea    0x9e71(%rip),%rcx        # 11400 <galois_ilog_tables>
    758f:	48 8b 14 d1          	mov    (%rcx,%rdx,8),%rdx
    7593:	48 98                	cltq   
    7595:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    7598:	c3                   	retq   
    7599:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    75a0:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    75a5:	c3                   	retq   
    75a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    75ad:	00 00 00 

00000000000075b0 <galois_create_mult_tables>:
    75b0:	f3 0f 1e fa          	endbr64 
    75b4:	83 ff 0d             	cmp    $0xd,%edi
    75b7:	0f 8f ed 01 00 00    	jg     77aa <galois_create_mult_tables+0x1fa>
    75bd:	41 57                	push   %r15
    75bf:	41 56                	push   %r14
    75c1:	4c 63 f7             	movslq %edi,%r14
    75c4:	41 55                	push   %r13
    75c6:	41 89 fd             	mov    %edi,%r13d
    75c9:	41 54                	push   %r12
    75cb:	55                   	push   %rbp
    75cc:	53                   	push   %rbx
    75cd:	48 8d 1d 0c 9d 00 00 	lea    0x9d0c(%rip),%rbx        # 112e0 <galois_mult_tables>
    75d4:	48 83 ec 18          	sub    $0x18,%rsp
    75d8:	4a 83 3c f3 00       	cmpq   $0x0,(%rbx,%r14,8)
    75dd:	74 11                	je     75f0 <galois_create_mult_tables+0x40>
    75df:	31 c0                	xor    %eax,%eax
    75e1:	48 83 c4 18          	add    $0x18,%rsp
    75e5:	5b                   	pop    %rbx
    75e6:	5d                   	pop    %rbp
    75e7:	41 5c                	pop    %r12
    75e9:	41 5d                	pop    %r13
    75eb:	41 5e                	pop    %r14
    75ed:	41 5f                	pop    %r15
    75ef:	c3                   	retq   
    75f0:	48 8d 05 c9 35 00 00 	lea    0x35c9(%rip),%rax        # abc0 <nw>
    75f7:	4e 63 0c b0          	movslq (%rax,%r14,4),%r9
    75fb:	4d 89 cc             	mov    %r9,%r12
    75fe:	4d 0f af e1          	imul   %r9,%r12
    7602:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
    7607:	4d 89 cf             	mov    %r9,%r15
    760a:	49 c1 e4 02          	shl    $0x2,%r12
    760e:	4c 89 e7             	mov    %r12,%rdi
    7611:	e8 ea 9d ff ff       	callq  1400 <malloc@plt>
    7616:	4a 89 04 f3          	mov    %rax,(%rbx,%r14,8)
    761a:	48 89 c5             	mov    %rax,%rbp
    761d:	48 85 c0             	test   %rax,%rax
    7620:	0f 84 62 01 00 00    	je     7788 <galois_create_mult_tables+0x1d8>
    7626:	4c 89 e7             	mov    %r12,%rdi
    7629:	e8 d2 9d ff ff       	callq  1400 <malloc@plt>
    762e:	48 85 c0             	test   %rax,%rax
    7631:	48 8d 0d 88 9b 00 00 	lea    0x9b88(%rip),%rcx        # 111c0 <galois_div_tables>
    7638:	4a 89 04 f1          	mov    %rax,(%rcx,%r14,8)
    763c:	4c 8b 4c 24 08       	mov    0x8(%rsp),%r9
    7641:	49 89 c4             	mov    %rax,%r12
    7644:	0f 84 48 01 00 00    	je     7792 <galois_create_mult_tables+0x1e2>
    764a:	48 8d 15 cf 9e 00 00 	lea    0x9ecf(%rip),%rdx        # 11520 <galois_log_tables>
    7651:	4a 83 3c f2 00       	cmpq   $0x0,(%rdx,%r14,8)
    7656:	0f 84 cb 00 00 00    	je     7727 <galois_create_mult_tables+0x177>
    765c:	c7 45 00 00 00 00 00 	movl   $0x0,0x0(%rbp)
    7663:	41 c7 04 24 ff ff ff 	movl   $0xffffffff,(%r12)
    766a:	ff 
    766b:	41 83 ff 01          	cmp    $0x1,%r15d
    766f:	0f 8e 6a ff ff ff    	jle    75df <galois_create_mult_tables+0x2f>
    7675:	41 8d 4f fe          	lea    -0x2(%r15),%ecx
    7679:	4c 8d 59 02          	lea    0x2(%rcx),%r11
    767d:	b8 01 00 00 00       	mov    $0x1,%eax
    7682:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7688:	c7 44 85 00 00 00 00 	movl   $0x0,0x0(%rbp,%rax,4)
    768f:	00 
    7690:	41 c7 04 84 00 00 00 	movl   $0x0,(%r12,%rax,4)
    7697:	00 
    7698:	48 ff c0             	inc    %rax
    769b:	4c 39 d8             	cmp    %r11,%rax
    769e:	75 e8                	jne    7688 <galois_create_mult_tables+0xd8>
    76a0:	4a 8b 3c f2          	mov    (%rdx,%r14,8),%rdi
    76a4:	48 8d 05 55 9d 00 00 	lea    0x9d55(%rip),%rax        # 11400 <galois_ilog_tables>
    76ab:	4e 8b 04 f0          	mov    (%rax,%r14,8),%r8
    76af:	48 8d 5f 04          	lea    0x4(%rdi),%rbx
    76b3:	4c 8d 74 8f 08       	lea    0x8(%rdi,%rcx,4),%r14
    76b8:	45 8d 6f ff          	lea    -0x1(%r15),%r13d
    76bc:	0f 1f 40 00          	nopl   0x0(%rax)
    76c0:	49 c1 e1 02          	shl    $0x2,%r9
    76c4:	4e 8d 54 0d 00       	lea    0x0(%rbp,%r9,1),%r10
    76c9:	4d 01 e1             	add    %r12,%r9
    76cc:	41 c7 02 00 00 00 00 	movl   $0x0,(%r10)
    76d3:	41 c7 01 ff ff ff ff 	movl   $0xffffffff,(%r9)
    76da:	41 8d 47 01          	lea    0x1(%r15),%eax
    76de:	ba 01 00 00 00       	mov    $0x1,%edx
    76e3:	8b 33                	mov    (%rbx),%esi
    76e5:	0f 1f 00             	nopl   (%rax)
    76e8:	8b 0c 97             	mov    (%rdi,%rdx,4),%ecx
    76eb:	01 f1                	add    %esi,%ecx
    76ed:	48 63 c9             	movslq %ecx,%rcx
    76f0:	41 8b 0c 88          	mov    (%r8,%rcx,4),%ecx
    76f4:	41 89 0c 92          	mov    %ecx,(%r10,%rdx,4)
    76f8:	89 f1                	mov    %esi,%ecx
    76fa:	2b 0c 97             	sub    (%rdi,%rdx,4),%ecx
    76fd:	48 63 c9             	movslq %ecx,%rcx
    7700:	41 8b 0c 88          	mov    (%r8,%rcx,4),%ecx
    7704:	41 89 0c 91          	mov    %ecx,(%r9,%rdx,4)
    7708:	48 ff c2             	inc    %rdx
    770b:	4c 39 da             	cmp    %r11,%rdx
    770e:	75 d8                	jne    76e8 <galois_create_mult_tables+0x138>
    7710:	48 83 c3 04          	add    $0x4,%rbx
    7714:	45 8d 7c 05 00       	lea    0x0(%r13,%rax,1),%r15d
    7719:	49 39 de             	cmp    %rbx,%r14
    771c:	0f 84 bd fe ff ff    	je     75df <galois_create_mult_tables+0x2f>
    7722:	4d 63 cf             	movslq %r15d,%r9
    7725:	eb 99                	jmp    76c0 <galois_create_mult_tables+0x110>
    7727:	44 89 ef             	mov    %r13d,%edi
    772a:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
    772f:	e8 2c fc ff ff       	callq  7360 <galois_create_log_tables.part.0>
    7734:	85 c0                	test   %eax,%eax
    7736:	48 8d 0d 83 9a 00 00 	lea    0x9a83(%rip),%rcx        # 111c0 <galois_div_tables>
    773d:	78 19                	js     7758 <galois_create_mult_tables+0x1a8>
    773f:	4a 8b 2c f3          	mov    (%rbx,%r14,8),%rbp
    7743:	4e 8b 24 f1          	mov    (%rcx,%r14,8),%r12
    7747:	4c 8b 4c 24 08       	mov    0x8(%rsp),%r9
    774c:	48 8d 15 cd 9d 00 00 	lea    0x9dcd(%rip),%rdx        # 11520 <galois_log_tables>
    7753:	e9 04 ff ff ff       	jmpq   765c <galois_create_mult_tables+0xac>
    7758:	4a 8b 3c f3          	mov    (%rbx,%r14,8),%rdi
    775c:	e8 1f 9b ff ff       	callq  1280 <free@plt>
    7761:	48 8d 0d 58 9a 00 00 	lea    0x9a58(%rip),%rcx        # 111c0 <galois_div_tables>
    7768:	4a 8b 3c f1          	mov    (%rcx,%r14,8),%rdi
    776c:	e8 0f 9b ff ff       	callq  1280 <free@plt>
    7771:	48 8d 0d 48 9a 00 00 	lea    0x9a48(%rip),%rcx        # 111c0 <galois_div_tables>
    7778:	4a c7 04 f3 00 00 00 	movq   $0x0,(%rbx,%r14,8)
    777f:	00 
    7780:	4a c7 04 f1 00 00 00 	movq   $0x0,(%rcx,%r14,8)
    7787:	00 
    7788:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    778d:	e9 4f fe ff ff       	jmpq   75e1 <galois_create_mult_tables+0x31>
    7792:	48 89 ef             	mov    %rbp,%rdi
    7795:	e8 e6 9a ff ff       	callq  1280 <free@plt>
    779a:	4a c7 04 f3 00 00 00 	movq   $0x0,(%rbx,%r14,8)
    77a1:	00 
    77a2:	83 c8 ff             	or     $0xffffffff,%eax
    77a5:	e9 37 fe ff ff       	jmpq   75e1 <galois_create_mult_tables+0x31>
    77aa:	83 c8 ff             	or     $0xffffffff,%eax
    77ad:	c3                   	retq   
    77ae:	66 90                	xchg   %ax,%ax

00000000000077b0 <galois_ilog>:
    77b0:	f3 0f 1e fa          	endbr64 
    77b4:	41 54                	push   %r12
    77b6:	4c 8d 25 43 9c 00 00 	lea    0x9c43(%rip),%r12        # 11400 <galois_ilog_tables>
    77bd:	55                   	push   %rbp
    77be:	48 63 ee             	movslq %esi,%rbp
    77c1:	49 8b 04 ec          	mov    (%r12,%rbp,8),%rax
    77c5:	53                   	push   %rbx
    77c6:	89 fb                	mov    %edi,%ebx
    77c8:	48 85 c0             	test   %rax,%rax
    77cb:	74 13                	je     77e0 <galois_ilog+0x30>
    77cd:	48 63 fb             	movslq %ebx,%rdi
    77d0:	5b                   	pop    %rbx
    77d1:	5d                   	pop    %rbp
    77d2:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    77d5:	41 5c                	pop    %r12
    77d7:	c3                   	retq   
    77d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    77df:	00 
    77e0:	83 fe 1e             	cmp    $0x1e,%esi
    77e3:	7f 1f                	jg     7804 <galois_ilog+0x54>
    77e5:	48 8d 15 34 9d 00 00 	lea    0x9d34(%rip),%rdx        # 11520 <galois_log_tables>
    77ec:	48 83 3c ea 00       	cmpq   $0x0,(%rdx,%rbp,8)
    77f1:	75 da                	jne    77cd <galois_ilog+0x1d>
    77f3:	89 f7                	mov    %esi,%edi
    77f5:	e8 66 fb ff ff       	callq  7360 <galois_create_log_tables.part.0>
    77fa:	85 c0                	test   %eax,%eax
    77fc:	78 06                	js     7804 <galois_ilog+0x54>
    77fe:	49 8b 04 ec          	mov    (%r12,%rbp,8),%rax
    7802:	eb c9                	jmp    77cd <galois_ilog+0x1d>
    7804:	48 8b 0d 35 99 00 00 	mov    0x9935(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    780b:	48 8d 3d c6 30 00 00 	lea    0x30c6(%rip),%rdi        # a8d8 <__PRETTY_FUNCTION__.5741+0x1ff>
    7812:	ba 2a 00 00 00       	mov    $0x2a,%edx
    7817:	be 01 00 00 00       	mov    $0x1,%esi
    781c:	e8 4f 9c ff ff       	callq  1470 <fwrite@plt>
    7821:	bf 01 00 00 00       	mov    $0x1,%edi
    7826:	e8 35 9c ff ff       	callq  1460 <exit@plt>
    782b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000007830 <galois_log>:
    7830:	f3 0f 1e fa          	endbr64 
    7834:	41 54                	push   %r12
    7836:	4c 63 e6             	movslq %esi,%r12
    7839:	55                   	push   %rbp
    783a:	48 8d 2d df 9c 00 00 	lea    0x9cdf(%rip),%rbp        # 11520 <galois_log_tables>
    7841:	4a 8b 44 e5 00       	mov    0x0(%rbp,%r12,8),%rax
    7846:	53                   	push   %rbx
    7847:	89 fb                	mov    %edi,%ebx
    7849:	48 85 c0             	test   %rax,%rax
    784c:	74 12                	je     7860 <galois_log+0x30>
    784e:	48 63 fb             	movslq %ebx,%rdi
    7851:	5b                   	pop    %rbx
    7852:	5d                   	pop    %rbp
    7853:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    7856:	41 5c                	pop    %r12
    7858:	c3                   	retq   
    7859:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7860:	83 fe 1e             	cmp    $0x1e,%esi
    7863:	7f 1b                	jg     7880 <galois_log+0x50>
    7865:	89 f7                	mov    %esi,%edi
    7867:	e8 f4 fa ff ff       	callq  7360 <galois_create_log_tables.part.0>
    786c:	85 c0                	test   %eax,%eax
    786e:	78 10                	js     7880 <galois_log+0x50>
    7870:	4a 8b 44 e5 00       	mov    0x0(%rbp,%r12,8),%rax
    7875:	48 63 fb             	movslq %ebx,%rdi
    7878:	5b                   	pop    %rbx
    7879:	5d                   	pop    %rbp
    787a:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    787d:	41 5c                	pop    %r12
    787f:	c3                   	retq   
    7880:	48 8b 0d b9 98 00 00 	mov    0x98b9(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    7887:	48 8d 3d 7a 30 00 00 	lea    0x307a(%rip),%rdi        # a908 <__PRETTY_FUNCTION__.5741+0x22f>
    788e:	ba 29 00 00 00       	mov    $0x29,%edx
    7893:	be 01 00 00 00       	mov    $0x1,%esi
    7898:	e8 d3 9b ff ff       	callq  1470 <fwrite@plt>
    789d:	bf 01 00 00 00       	mov    $0x1,%edi
    78a2:	e8 b9 9b ff ff       	callq  1460 <exit@plt>
    78a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    78ae:	00 00 

00000000000078b0 <galois_shift_multiply>:
    78b0:	f3 0f 1e fa          	endbr64 
    78b4:	41 54                	push   %r12
    78b6:	55                   	push   %rbp
    78b7:	53                   	push   %rbx
    78b8:	48 81 ec 90 00 00 00 	sub    $0x90,%rsp
    78bf:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    78c6:	00 00 
    78c8:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
    78cf:	00 
    78d0:	31 c0                	xor    %eax,%eax
    78d2:	85 d2                	test   %edx,%edx
    78d4:	0f 8e c1 00 00 00    	jle    799b <galois_shift_multiply+0xeb>
    78da:	44 8d 5a ff          	lea    -0x1(%rdx),%r11d
    78de:	48 89 e3             	mov    %rsp,%rbx
    78e1:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    78e7:	48 89 d8             	mov    %rbx,%rax
    78ea:	4e 8d 4c 9c 04       	lea    0x4(%rsp,%r11,4),%r9
    78ef:	4c 8d 25 0a 34 00 00 	lea    0x340a(%rip),%r12        # ad00 <prim_poly>
    78f6:	4c 63 d2             	movslq %edx,%r10
    78f9:	48 8d 2d 20 32 00 00 	lea    0x3220(%rip),%rbp        # ab20 <nwm1>
    7900:	c4 42 21 f7 c0       	shlx   %r11d,%r8d,%r8d
    7905:	0f 1f 00             	nopl   (%rax)
    7908:	44 89 c1             	mov    %r8d,%ecx
    790b:	21 f1                	and    %esi,%ecx
    790d:	89 30                	mov    %esi,(%rax)
    790f:	01 f6                	add    %esi,%esi
    7911:	85 c9                	test   %ecx,%ecx
    7913:	74 09                	je     791e <galois_shift_multiply+0x6e>
    7915:	43 33 34 94          	xor    (%r12,%r10,4),%esi
    7919:	42 23 74 95 00       	and    0x0(%rbp,%r10,4),%esi
    791e:	48 83 c0 04          	add    $0x4,%rax
    7922:	4c 39 c8             	cmp    %r9,%rax
    7925:	75 e1                	jne    7908 <galois_shift_multiply+0x58>
    7927:	45 31 c9             	xor    %r9d,%r9d
    792a:	45 31 c0             	xor    %r8d,%r8d
    792d:	bd 01 00 00 00       	mov    $0x1,%ebp
    7932:	eb 10                	jmp    7944 <galois_shift_multiply+0x94>
    7934:	0f 1f 40 00          	nopl   0x0(%rax)
    7938:	49 8d 41 01          	lea    0x1(%r9),%rax
    793c:	4d 39 d9             	cmp    %r11,%r9
    793f:	74 38                	je     7979 <galois_shift_multiply+0xc9>
    7941:	49 89 c1             	mov    %rax,%r9
    7944:	c4 e2 31 f7 c5       	shlx   %r9d,%ebp,%eax
    7949:	85 f8                	test   %edi,%eax
    794b:	74 eb                	je     7938 <galois_shift_multiply+0x88>
    794d:	46 8b 14 8b          	mov    (%rbx,%r9,4),%r10d
    7951:	31 c9                	xor    %ecx,%ecx
    7953:	b8 01 00 00 00       	mov    $0x1,%eax
    7958:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    795f:	00 
    7960:	44 89 d6             	mov    %r10d,%esi
    7963:	21 c6                	and    %eax,%esi
    7965:	ff c1                	inc    %ecx
    7967:	41 31 f0             	xor    %esi,%r8d
    796a:	01 c0                	add    %eax,%eax
    796c:	39 ca                	cmp    %ecx,%edx
    796e:	75 f0                	jne    7960 <galois_shift_multiply+0xb0>
    7970:	49 8d 41 01          	lea    0x1(%r9),%rax
    7974:	4d 39 d9             	cmp    %r11,%r9
    7977:	75 c8                	jne    7941 <galois_shift_multiply+0x91>
    7979:	48 8b 84 24 88 00 00 	mov    0x88(%rsp),%rax
    7980:	00 
    7981:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    7988:	00 00 
    798a:	75 14                	jne    79a0 <galois_shift_multiply+0xf0>
    798c:	48 81 c4 90 00 00 00 	add    $0x90,%rsp
    7993:	5b                   	pop    %rbx
    7994:	5d                   	pop    %rbp
    7995:	44 89 c0             	mov    %r8d,%eax
    7998:	41 5c                	pop    %r12
    799a:	c3                   	retq   
    799b:	45 31 c0             	xor    %r8d,%r8d
    799e:	eb d9                	jmp    7979 <galois_shift_multiply+0xc9>
    79a0:	e8 6b 99 ff ff       	callq  1310 <__stack_chk_fail@plt>
    79a5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    79ac:	00 00 00 00 

00000000000079b0 <galois_multtable_multiply>:
    79b0:	f3 0f 1e fa          	endbr64 
    79b4:	48 63 ca             	movslq %edx,%rcx
    79b7:	48 8d 05 22 99 00 00 	lea    0x9922(%rip),%rax        # 112e0 <galois_mult_tables>
    79be:	48 8b 04 c8          	mov    (%rax,%rcx,8),%rax
    79c2:	c4 e2 69 f7 ff       	shlx   %edx,%edi,%edi
    79c7:	09 f7                	or     %esi,%edi
    79c9:	48 63 ff             	movslq %edi,%rdi
    79cc:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    79cf:	c3                   	retq   

00000000000079d0 <galois_multtable_divide>:
    79d0:	f3 0f 1e fa          	endbr64 
    79d4:	48 63 ca             	movslq %edx,%rcx
    79d7:	48 8d 05 e2 97 00 00 	lea    0x97e2(%rip),%rax        # 111c0 <galois_div_tables>
    79de:	48 8b 04 c8          	mov    (%rax,%rcx,8),%rax
    79e2:	c4 e2 69 f7 ff       	shlx   %edx,%edi,%edi
    79e7:	09 f7                	or     %esi,%edi
    79e9:	48 63 ff             	movslq %edi,%rdi
    79ec:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    79ef:	c3                   	retq   

00000000000079f0 <galois_w08_region_multiply>:
    79f0:	f3 0f 1e fa          	endbr64 
    79f4:	41 54                	push   %r12
    79f6:	55                   	push   %rbp
    79f7:	89 d5                	mov    %edx,%ebp
    79f9:	53                   	push   %rbx
    79fa:	48 89 fb             	mov    %rdi,%rbx
    79fd:	48 83 ec 20          	sub    $0x20,%rsp
    7a01:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    7a08:	00 00 
    7a0a:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    7a0f:	31 c0                	xor    %eax,%eax
    7a11:	48 8b 05 08 99 00 00 	mov    0x9908(%rip),%rax        # 11320 <galois_mult_tables+0x40>
    7a18:	48 85 c9             	test   %rcx,%rcx
    7a1b:	0f 84 83 00 00 00    	je     7aa4 <galois_w08_region_multiply+0xb4>
    7a21:	49 89 cc             	mov    %rcx,%r12
    7a24:	48 85 c0             	test   %rax,%rax
    7a27:	0f 84 c0 00 00 00    	je     7aed <galois_w08_region_multiply+0xfd>
    7a2d:	c1 e6 08             	shl    $0x8,%esi
    7a30:	41 89 f1             	mov    %esi,%r9d
    7a33:	45 85 c0             	test   %r8d,%r8d
    7a36:	74 7e                	je     7ab6 <galois_w08_region_multiply+0xc6>
    7a38:	85 ed                	test   %ebp,%ebp
    7a3a:	7e 4b                	jle    7a87 <galois_w08_region_multiply+0x97>
    7a3c:	4c 8b 15 dd 98 00 00 	mov    0x98dd(%rip),%r10        # 11320 <galois_mult_tables+0x40>
    7a43:	48 89 df             	mov    %rbx,%rdi
    7a46:	31 d2                	xor    %edx,%edx
    7a48:	4c 8d 44 24 10       	lea    0x10(%rsp),%r8
    7a4d:	0f 1f 00             	nopl   (%rax)
    7a50:	31 f6                	xor    %esi,%esi
    7a52:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7a58:	0f b6 04 37          	movzbl (%rdi,%rsi,1),%eax
    7a5c:	44 01 c8             	add    %r9d,%eax
    7a5f:	48 98                	cltq   
    7a61:	41 8b 04 82          	mov    (%r10,%rax,4),%eax
    7a65:	41 88 04 30          	mov    %al,(%r8,%rsi,1)
    7a69:	48 ff c6             	inc    %rsi
    7a6c:	48 83 fe 08          	cmp    $0x8,%rsi
    7a70:	75 e6                	jne    7a58 <galois_w08_region_multiply+0x68>
    7a72:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    7a77:	48 83 c7 08          	add    $0x8,%rdi
    7a7b:	49 31 04 14          	xor    %rax,(%r12,%rdx,1)
    7a7f:	48 83 c2 08          	add    $0x8,%rdx
    7a83:	39 d5                	cmp    %edx,%ebp
    7a85:	7f c9                	jg     7a50 <galois_w08_region_multiply+0x60>
    7a87:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    7a8c:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    7a93:	00 00 
    7a95:	0f 85 c1 00 00 00    	jne    7b5c <galois_w08_region_multiply+0x16c>
    7a9b:	48 83 c4 20          	add    $0x20,%rsp
    7a9f:	5b                   	pop    %rbx
    7aa0:	5d                   	pop    %rbp
    7aa1:	41 5c                	pop    %r12
    7aa3:	c3                   	retq   
    7aa4:	48 85 c0             	test   %rax,%rax
    7aa7:	0f 84 93 00 00 00    	je     7b40 <galois_w08_region_multiply+0x150>
    7aad:	c1 e6 08             	shl    $0x8,%esi
    7ab0:	41 89 f1             	mov    %esi,%r9d
    7ab3:	49 89 dc             	mov    %rbx,%r12
    7ab6:	85 ed                	test   %ebp,%ebp
    7ab8:	7e cd                	jle    7a87 <galois_w08_region_multiply+0x97>
    7aba:	48 8b 35 5f 98 00 00 	mov    0x985f(%rip),%rsi        # 11320 <galois_mult_tables+0x40>
    7ac1:	8d 4d ff             	lea    -0x1(%rbp),%ecx
    7ac4:	31 d2                	xor    %edx,%edx
    7ac6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    7acd:	00 00 00 
    7ad0:	0f b6 04 13          	movzbl (%rbx,%rdx,1),%eax
    7ad4:	44 01 c8             	add    %r9d,%eax
    7ad7:	48 98                	cltq   
    7ad9:	8b 04 86             	mov    (%rsi,%rax,4),%eax
    7adc:	41 88 04 14          	mov    %al,(%r12,%rdx,1)
    7ae0:	48 89 d0             	mov    %rdx,%rax
    7ae3:	48 ff c2             	inc    %rdx
    7ae6:	48 39 c8             	cmp    %rcx,%rax
    7ae9:	75 e5                	jne    7ad0 <galois_w08_region_multiply+0xe0>
    7aeb:	eb 9a                	jmp    7a87 <galois_w08_region_multiply+0x97>
    7aed:	bf 08 00 00 00       	mov    $0x8,%edi
    7af2:	44 89 44 24 0c       	mov    %r8d,0xc(%rsp)
    7af7:	89 74 24 08          	mov    %esi,0x8(%rsp)
    7afb:	e8 b0 fa ff ff       	callq  75b0 <galois_create_mult_tables>
    7b00:	85 c0                	test   %eax,%eax
    7b02:	8b 74 24 08          	mov    0x8(%rsp),%esi
    7b06:	44 8b 44 24 0c       	mov    0xc(%rsp),%r8d
    7b0b:	0f 89 1c ff ff ff    	jns    7a2d <galois_w08_region_multiply+0x3d>
    7b11:	48 8b 0d 28 96 00 00 	mov    0x9628(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    7b18:	48 8d 3d 19 2e 00 00 	lea    0x2e19(%rip),%rdi        # a938 <__PRETTY_FUNCTION__.5741+0x25f>
    7b1f:	ba 41 00 00 00       	mov    $0x41,%edx
    7b24:	be 01 00 00 00       	mov    $0x1,%esi
    7b29:	e8 42 99 ff ff       	callq  1470 <fwrite@plt>
    7b2e:	bf 01 00 00 00       	mov    $0x1,%edi
    7b33:	e8 28 99 ff ff       	callq  1460 <exit@plt>
    7b38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    7b3f:	00 
    7b40:	bf 08 00 00 00       	mov    $0x8,%edi
    7b45:	89 74 24 08          	mov    %esi,0x8(%rsp)
    7b49:	e8 62 fa ff ff       	callq  75b0 <galois_create_mult_tables>
    7b4e:	85 c0                	test   %eax,%eax
    7b50:	8b 74 24 08          	mov    0x8(%rsp),%esi
    7b54:	0f 89 53 ff ff ff    	jns    7aad <galois_w08_region_multiply+0xbd>
    7b5a:	eb b5                	jmp    7b11 <galois_w08_region_multiply+0x121>
    7b5c:	e8 af 97 ff ff       	callq  1310 <__stack_chk_fail@plt>
    7b61:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    7b68:	00 00 00 00 
    7b6c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000007b70 <galois_w16_region_multiply>:
    7b70:	f3 0f 1e fa          	endbr64 
    7b74:	41 55                	push   %r13
    7b76:	41 54                	push   %r12
    7b78:	41 89 d4             	mov    %edx,%r12d
    7b7b:	55                   	push   %rbp
    7b7c:	48 89 cd             	mov    %rcx,%rbp
    7b7f:	53                   	push   %rbx
    7b80:	48 83 ec 38          	sub    $0x38,%rsp
    7b84:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    7b8b:	00 00 
    7b8d:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    7b92:	31 c0                	xor    %eax,%eax
    7b94:	48 85 c9             	test   %rcx,%rcx
    7b97:	48 0f 44 ef          	cmove  %rdi,%rbp
    7b9b:	41 c1 ec 1f          	shr    $0x1f,%r12d
    7b9f:	41 01 d4             	add    %edx,%r12d
    7ba2:	41 d1 fc             	sar    %r12d
    7ba5:	85 f6                	test   %esi,%esi
    7ba7:	0f 84 f3 00 00 00    	je     7ca0 <galois_w16_region_multiply+0x130>
    7bad:	4c 8b 0d ec 99 00 00 	mov    0x99ec(%rip),%r9        # 115a0 <galois_log_tables+0x80>
    7bb4:	48 89 fb             	mov    %rdi,%rbx
    7bb7:	4d 85 c9             	test   %r9,%r9
    7bba:	0f 84 16 01 00 00    	je     7cd6 <galois_w16_region_multiply+0x166>
    7bc0:	4c 63 ee             	movslq %esi,%r13
    7bc3:	47 8b 14 a9          	mov    (%r9,%r13,4),%r10d
    7bc7:	48 85 c9             	test   %rcx,%rcx
    7bca:	74 74                	je     7c40 <galois_w16_region_multiply+0xd0>
    7bcc:	45 85 c0             	test   %r8d,%r8d
    7bcf:	74 6f                	je     7c40 <galois_w16_region_multiply+0xd0>
    7bd1:	83 fa 01             	cmp    $0x1,%edx
    7bd4:	7e 4b                	jle    7c21 <galois_w16_region_multiply+0xb1>
    7bd6:	4c 8b 05 a3 98 00 00 	mov    0x98a3(%rip),%r8        # 11480 <galois_ilog_tables+0x80>
    7bdd:	4c 8b 5c 24 20       	mov    0x20(%rsp),%r11
    7be2:	48 89 df             	mov    %rbx,%rdi
    7be5:	31 c9                	xor    %ecx,%ecx
    7be7:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
    7bec:	0f 1f 40 00          	nopl   0x0(%rax)
    7bf0:	31 c0                	xor    %eax,%eax
    7bf2:	0f b7 14 07          	movzwl (%rdi,%rax,1),%edx
    7bf6:	66 85 d2             	test   %dx,%dx
    7bf9:	0f 85 81 00 00 00    	jne    7c80 <galois_w16_region_multiply+0x110>
    7bff:	31 d2                	xor    %edx,%edx
    7c01:	66 89 14 06          	mov    %dx,(%rsi,%rax,1)
    7c05:	48 83 c0 02          	add    $0x2,%rax
    7c09:	48 83 f8 08          	cmp    $0x8,%rax
    7c0d:	75 e3                	jne    7bf2 <galois_w16_region_multiply+0x82>
    7c0f:	4c 31 5c 4d 00       	xor    %r11,0x0(%rbp,%rcx,2)
    7c14:	48 83 c1 04          	add    $0x4,%rcx
    7c18:	48 83 c7 08          	add    $0x8,%rdi
    7c1c:	41 39 cc             	cmp    %ecx,%r12d
    7c1f:	7f cf                	jg     7bf0 <galois_w16_region_multiply+0x80>
    7c21:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    7c26:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    7c2d:	00 00 
    7c2f:	0f 85 06 01 00 00    	jne    7d3b <galois_w16_region_multiply+0x1cb>
    7c35:	48 83 c4 38          	add    $0x38,%rsp
    7c39:	5b                   	pop    %rbx
    7c3a:	5d                   	pop    %rbp
    7c3b:	41 5c                	pop    %r12
    7c3d:	41 5d                	pop    %r13
    7c3f:	c3                   	retq   
    7c40:	83 fa 01             	cmp    $0x1,%edx
    7c43:	7e dc                	jle    7c21 <galois_w16_region_multiply+0xb1>
    7c45:	48 8b 0d 34 98 00 00 	mov    0x9834(%rip),%rcx        # 11480 <galois_ilog_tables+0x80>
    7c4c:	31 c0                	xor    %eax,%eax
    7c4e:	eb 0f                	jmp    7c5f <galois_w16_region_multiply+0xef>
    7c50:	31 f6                	xor    %esi,%esi
    7c52:	66 89 74 45 00       	mov    %si,0x0(%rbp,%rax,2)
    7c57:	48 ff c0             	inc    %rax
    7c5a:	41 39 c4             	cmp    %eax,%r12d
    7c5d:	7e c2                	jle    7c21 <galois_w16_region_multiply+0xb1>
    7c5f:	0f b7 14 43          	movzwl (%rbx,%rax,2),%edx
    7c63:	66 85 d2             	test   %dx,%dx
    7c66:	74 e8                	je     7c50 <galois_w16_region_multiply+0xe0>
    7c68:	41 8b 3c 91          	mov    (%r9,%rdx,4),%edi
    7c6c:	44 01 d7             	add    %r10d,%edi
    7c6f:	48 63 d7             	movslq %edi,%rdx
    7c72:	8b 14 91             	mov    (%rcx,%rdx,4),%edx
    7c75:	66 89 54 45 00       	mov    %dx,0x0(%rbp,%rax,2)
    7c7a:	eb db                	jmp    7c57 <galois_w16_region_multiply+0xe7>
    7c7c:	0f 1f 40 00          	nopl   0x0(%rax)
    7c80:	41 8b 1c 91          	mov    (%r9,%rdx,4),%ebx
    7c84:	44 01 d3             	add    %r10d,%ebx
    7c87:	48 63 d3             	movslq %ebx,%rdx
    7c8a:	41 8b 14 90          	mov    (%r8,%rdx,4),%edx
    7c8e:	66 89 14 06          	mov    %dx,(%rsi,%rax,1)
    7c92:	e9 6e ff ff ff       	jmpq   7c05 <galois_w16_region_multiply+0x95>
    7c97:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    7c9e:	00 00 
    7ca0:	45 85 c0             	test   %r8d,%r8d
    7ca3:	0f 85 78 ff ff ff    	jne    7c21 <galois_w16_region_multiply+0xb1>
    7ca9:	4d 63 e4             	movslq %r12d,%r12
    7cac:	4a 8d 44 65 00       	lea    0x0(%rbp,%r12,2),%rax
    7cb1:	48 39 c5             	cmp    %rax,%rbp
    7cb4:	0f 83 67 ff ff ff    	jae    7c21 <galois_w16_region_multiply+0xb1>
    7cba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7cc0:	48 c7 45 00 00 00 00 	movq   $0x0,0x0(%rbp)
    7cc7:	00 
    7cc8:	48 83 c5 08          	add    $0x8,%rbp
    7ccc:	48 39 e8             	cmp    %rbp,%rax
    7ccf:	77 ef                	ja     7cc0 <galois_w16_region_multiply+0x150>
    7cd1:	e9 4b ff ff ff       	jmpq   7c21 <galois_w16_region_multiply+0xb1>
    7cd6:	bf 10 00 00 00       	mov    $0x10,%edi
    7cdb:	44 89 44 24 1c       	mov    %r8d,0x1c(%rsp)
    7ce0:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    7ce5:	89 54 24 18          	mov    %edx,0x18(%rsp)
    7ce9:	89 74 24 0c          	mov    %esi,0xc(%rsp)
    7ced:	e8 6e f6 ff ff       	callq  7360 <galois_create_log_tables.part.0>
    7cf2:	85 c0                	test   %eax,%eax
    7cf4:	78 1e                	js     7d14 <galois_w16_region_multiply+0x1a4>
    7cf6:	4c 8b 0d a3 98 00 00 	mov    0x98a3(%rip),%r9        # 115a0 <galois_log_tables+0x80>
    7cfd:	8b 74 24 0c          	mov    0xc(%rsp),%esi
    7d01:	8b 54 24 18          	mov    0x18(%rsp),%edx
    7d05:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    7d0a:	44 8b 44 24 1c       	mov    0x1c(%rsp),%r8d
    7d0f:	e9 ac fe ff ff       	jmpq   7bc0 <galois_w16_region_multiply+0x50>
    7d14:	48 8b 0d 25 94 00 00 	mov    0x9425(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    7d1b:	48 8d 3d 5e 2c 00 00 	lea    0x2c5e(%rip),%rdi        # a980 <__PRETTY_FUNCTION__.5741+0x2a7>
    7d22:	ba 36 00 00 00       	mov    $0x36,%edx
    7d27:	be 01 00 00 00       	mov    $0x1,%esi
    7d2c:	e8 3f 97 ff ff       	callq  1470 <fwrite@plt>
    7d31:	bf 01 00 00 00       	mov    $0x1,%edi
    7d36:	e8 25 97 ff ff       	callq  1460 <exit@plt>
    7d3b:	e8 d0 95 ff ff       	callq  1310 <__stack_chk_fail@plt>

0000000000007d40 <galois_invert_binary_matrix>:
    7d40:	f3 0f 1e fa          	endbr64 
    7d44:	85 d2                	test   %edx,%edx
    7d46:	0f 8e 58 01 00 00    	jle    7ea4 <galois_invert_binary_matrix+0x164>
    7d4c:	41 55                	push   %r13
    7d4e:	44 8d 5a ff          	lea    -0x1(%rdx),%r11d
    7d52:	41 89 d1             	mov    %edx,%r9d
    7d55:	41 54                	push   %r12
    7d57:	4d 89 da             	mov    %r11,%r10
    7d5a:	31 c0                	xor    %eax,%eax
    7d5c:	55                   	push   %rbp
    7d5d:	b9 01 00 00 00       	mov    $0x1,%ecx
    7d62:	49 8d 6b 01          	lea    0x1(%r11),%rbp
    7d66:	53                   	push   %rbx
    7d67:	48 83 ec 08          	sub    $0x8,%rsp
    7d6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7d70:	c4 e2 79 f7 d1       	shlx   %eax,%ecx,%edx
    7d75:	89 14 86             	mov    %edx,(%rsi,%rax,4)
    7d78:	48 89 c2             	mov    %rax,%rdx
    7d7b:	48 ff c0             	inc    %rax
    7d7e:	4c 39 da             	cmp    %r11,%rdx
    7d81:	75 ed                	jne    7d70 <galois_invert_binary_matrix+0x30>
    7d83:	49 83 c3 02          	add    $0x2,%r11
    7d87:	b9 01 00 00 00       	mov    $0x1,%ecx
    7d8c:	bb 01 00 00 00       	mov    $0x1,%ebx
    7d91:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7d98:	44 8b 6c 8f fc       	mov    -0x4(%rdi,%rcx,4),%r13d
    7d9d:	44 8d 41 ff          	lea    -0x1(%rcx),%r8d
    7da1:	45 0f a3 c5          	bt     %r8d,%r13d
    7da5:	89 ca                	mov    %ecx,%edx
    7da7:	0f 83 9b 00 00 00    	jae    7e48 <galois_invert_binary_matrix+0x108>
    7dad:	48 39 e9             	cmp    %rbp,%rcx
    7db0:	74 3e                	je     7df0 <galois_invert_binary_matrix+0xb0>
    7db2:	48 89 c8             	mov    %rcx,%rax
    7db5:	c4 62 39 f7 c3       	shlx   %r8d,%ebx,%r8d
    7dba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7dc0:	8b 14 87             	mov    (%rdi,%rax,4),%edx
    7dc3:	44 85 c2             	test   %r8d,%edx
    7dc6:	74 0e                	je     7dd6 <galois_invert_binary_matrix+0x96>
    7dc8:	33 54 8f fc          	xor    -0x4(%rdi,%rcx,4),%edx
    7dcc:	89 14 87             	mov    %edx,(%rdi,%rax,4)
    7dcf:	8b 54 8e fc          	mov    -0x4(%rsi,%rcx,4),%edx
    7dd3:	31 14 86             	xor    %edx,(%rsi,%rax,4)
    7dd6:	48 ff c0             	inc    %rax
    7dd9:	41 39 c1             	cmp    %eax,%r9d
    7ddc:	75 e2                	jne    7dc0 <galois_invert_binary_matrix+0x80>
    7dde:	48 ff c1             	inc    %rcx
    7de1:	49 39 cb             	cmp    %rcx,%r11
    7de4:	75 b2                	jne    7d98 <galois_invert_binary_matrix+0x58>
    7de6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    7ded:	00 00 00 
    7df0:	4d 63 d2             	movslq %r10d,%r10
    7df3:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    7df9:	45 85 d2             	test   %r10d,%r10d
    7dfc:	7e 33                	jle    7e31 <galois_invert_binary_matrix+0xf1>
    7dfe:	41 8d 4a ff          	lea    -0x1(%r10),%ecx
    7e02:	31 c0                	xor    %eax,%eax
    7e04:	c4 42 29 f7 c1       	shlx   %r10d,%r9d,%r8d
    7e09:	eb 08                	jmp    7e13 <galois_invert_binary_matrix+0xd3>
    7e0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7e10:	48 89 d0             	mov    %rdx,%rax
    7e13:	44 85 04 87          	test   %r8d,(%rdi,%rax,4)
    7e17:	74 07                	je     7e20 <galois_invert_binary_matrix+0xe0>
    7e19:	42 8b 14 96          	mov    (%rsi,%r10,4),%edx
    7e1d:	31 14 86             	xor    %edx,(%rsi,%rax,4)
    7e20:	48 8d 50 01          	lea    0x1(%rax),%rdx
    7e24:	48 39 c8             	cmp    %rcx,%rax
    7e27:	75 e7                	jne    7e10 <galois_invert_binary_matrix+0xd0>
    7e29:	49 ff ca             	dec    %r10
    7e2c:	45 85 d2             	test   %r10d,%r10d
    7e2f:	7f cd                	jg     7dfe <galois_invert_binary_matrix+0xbe>
    7e31:	44 89 d0             	mov    %r10d,%eax
    7e34:	ff c8                	dec    %eax
    7e36:	79 f1                	jns    7e29 <galois_invert_binary_matrix+0xe9>
    7e38:	48 83 c4 08          	add    $0x8,%rsp
    7e3c:	5b                   	pop    %rbx
    7e3d:	5d                   	pop    %rbp
    7e3e:	41 5c                	pop    %r12
    7e40:	41 5d                	pop    %r13
    7e42:	c3                   	retq   
    7e43:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7e48:	41 39 c9             	cmp    %ecx,%r9d
    7e4b:	7e 26                	jle    7e73 <galois_invert_binary_matrix+0x133>
    7e4d:	48 89 c8             	mov    %rcx,%rax
    7e50:	c4 62 39 f7 e3       	shlx   %r8d,%ebx,%r12d
    7e55:	eb 14                	jmp    7e6b <galois_invert_binary_matrix+0x12b>
    7e57:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    7e5e:	00 00 
    7e60:	8d 50 01             	lea    0x1(%rax),%edx
    7e63:	48 ff c0             	inc    %rax
    7e66:	41 39 c1             	cmp    %eax,%r9d
    7e69:	7e 08                	jle    7e73 <galois_invert_binary_matrix+0x133>
    7e6b:	89 c2                	mov    %eax,%edx
    7e6d:	44 85 24 87          	test   %r12d,(%rdi,%rax,4)
    7e71:	74 ed                	je     7e60 <galois_invert_binary_matrix+0x120>
    7e73:	41 39 d1             	cmp    %edx,%r9d
    7e76:	74 2d                	je     7ea5 <galois_invert_binary_matrix+0x165>
    7e78:	48 63 c2             	movslq %edx,%rax
    7e7b:	48 c1 e0 02          	shl    $0x2,%rax
    7e7f:	48 8d 14 07          	lea    (%rdi,%rax,1),%rdx
    7e83:	44 8b 22             	mov    (%rdx),%r12d
    7e86:	48 01 f0             	add    %rsi,%rax
    7e89:	44 89 64 8f fc       	mov    %r12d,-0x4(%rdi,%rcx,4)
    7e8e:	44 89 2a             	mov    %r13d,(%rdx)
    7e91:	8b 54 8e fc          	mov    -0x4(%rsi,%rcx,4),%edx
    7e95:	44 8b 20             	mov    (%rax),%r12d
    7e98:	44 89 64 8e fc       	mov    %r12d,-0x4(%rsi,%rcx,4)
    7e9d:	89 10                	mov    %edx,(%rax)
    7e9f:	e9 09 ff ff ff       	jmpq   7dad <galois_invert_binary_matrix+0x6d>
    7ea4:	c3                   	retq   
    7ea5:	48 8b 0d 94 92 00 00 	mov    0x9294(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    7eac:	48 8d 3d 05 2b 00 00 	lea    0x2b05(%rip),%rdi        # a9b8 <__PRETTY_FUNCTION__.5741+0x2df>
    7eb3:	ba 2e 00 00 00       	mov    $0x2e,%edx
    7eb8:	be 01 00 00 00       	mov    $0x1,%esi
    7ebd:	e8 ae 95 ff ff       	callq  1470 <fwrite@plt>
    7ec2:	bf 01 00 00 00       	mov    $0x1,%edi
    7ec7:	e8 94 95 ff ff       	callq  1460 <exit@plt>
    7ecc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000007ed0 <galois_shift_inverse>:
    7ed0:	f3 0f 1e fa          	endbr64 
    7ed4:	55                   	push   %rbp
    7ed5:	89 f2                	mov    %esi,%edx
    7ed7:	48 81 ec 10 01 00 00 	sub    $0x110,%rsp
    7ede:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    7ee5:	00 00 
    7ee7:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
    7eee:	00 
    7eef:	31 c0                	xor    %eax,%eax
    7ef1:	85 f6                	test   %esi,%esi
    7ef3:	0f 8e 8f 00 00 00    	jle    7f88 <galois_shift_inverse+0xb8>
    7ef9:	8d 4e ff             	lea    -0x1(%rsi),%ecx
    7efc:	48 63 f1             	movslq %ecx,%rsi
    7eff:	48 8d 05 ba 2c 00 00 	lea    0x2cba(%rip),%rax        # abc0 <nw>
    7f06:	48 89 e5             	mov    %rsp,%rbp
    7f09:	89 c9                	mov    %ecx,%ecx
    7f0b:	44 8b 14 b0          	mov    (%rax,%rsi,4),%r10d
    7f0f:	4c 8d 4c 8c 04       	lea    0x4(%rsp,%rcx,4),%r9
    7f14:	48 89 e8             	mov    %rbp,%rax
    7f17:	4c 8d 1d e2 2d 00 00 	lea    0x2de2(%rip),%r11        # ad00 <prim_poly>
    7f1e:	4c 63 c2             	movslq %edx,%r8
    7f21:	48 8d 35 f8 2b 00 00 	lea    0x2bf8(%rip),%rsi        # ab20 <nwm1>
    7f28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    7f2f:	00 
    7f30:	44 89 d1             	mov    %r10d,%ecx
    7f33:	21 f9                	and    %edi,%ecx
    7f35:	89 38                	mov    %edi,(%rax)
    7f37:	01 ff                	add    %edi,%edi
    7f39:	85 c9                	test   %ecx,%ecx
    7f3b:	74 08                	je     7f45 <galois_shift_inverse+0x75>
    7f3d:	43 33 3c 83          	xor    (%r11,%r8,4),%edi
    7f41:	42 23 3c 86          	and    (%rsi,%r8,4),%edi
    7f45:	48 83 c0 04          	add    $0x4,%rax
    7f49:	4c 39 c8             	cmp    %r9,%rax
    7f4c:	75 e2                	jne    7f30 <galois_shift_inverse+0x60>
    7f4e:	48 8d b4 24 80 00 00 	lea    0x80(%rsp),%rsi
    7f55:	00 
    7f56:	48 89 ef             	mov    %rbp,%rdi
    7f59:	e8 e2 fd ff ff       	callq  7d40 <galois_invert_binary_matrix>
    7f5e:	48 8b 94 24 08 01 00 	mov    0x108(%rsp),%rdx
    7f65:	00 
    7f66:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
    7f6d:	00 00 
    7f6f:	8b 84 24 80 00 00 00 	mov    0x80(%rsp),%eax
    7f76:	75 15                	jne    7f8d <galois_shift_inverse+0xbd>
    7f78:	48 81 c4 10 01 00 00 	add    $0x110,%rsp
    7f7f:	5d                   	pop    %rbp
    7f80:	c3                   	retq   
    7f81:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7f88:	48 89 e5             	mov    %rsp,%rbp
    7f8b:	eb c1                	jmp    7f4e <galois_shift_inverse+0x7e>
    7f8d:	e8 7e 93 ff ff       	callq  1310 <__stack_chk_fail@plt>
    7f92:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    7f99:	00 00 00 00 
    7f9d:	0f 1f 00             	nopl   (%rax)

0000000000007fa0 <galois_shift_divide>:
    7fa0:	f3 0f 1e fa          	endbr64 
    7fa4:	41 54                	push   %r12
    7fa6:	48 83 ec 10          	sub    $0x10,%rsp
    7faa:	85 f6                	test   %esi,%esi
    7fac:	74 3a                	je     7fe8 <galois_shift_divide+0x48>
    7fae:	41 89 fc             	mov    %edi,%r12d
    7fb1:	85 ff                	test   %edi,%edi
    7fb3:	75 0b                	jne    7fc0 <galois_shift_divide+0x20>
    7fb5:	48 83 c4 10          	add    $0x10,%rsp
    7fb9:	44 89 e0             	mov    %r12d,%eax
    7fbc:	41 5c                	pop    %r12
    7fbe:	c3                   	retq   
    7fbf:	90                   	nop
    7fc0:	89 f7                	mov    %esi,%edi
    7fc2:	89 d6                	mov    %edx,%esi
    7fc4:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    7fc8:	e8 03 ff ff ff       	callq  7ed0 <galois_shift_inverse>
    7fcd:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    7fd1:	48 83 c4 10          	add    $0x10,%rsp
    7fd5:	44 89 e7             	mov    %r12d,%edi
    7fd8:	89 c6                	mov    %eax,%esi
    7fda:	41 5c                	pop    %r12
    7fdc:	e9 cf f8 ff ff       	jmpq   78b0 <galois_shift_multiply>
    7fe1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7fe8:	41 bc ff ff ff ff    	mov    $0xffffffff,%r12d
    7fee:	eb c5                	jmp    7fb5 <galois_shift_divide+0x15>

0000000000007ff0 <galois_get_mult_table>:
    7ff0:	f3 0f 1e fa          	endbr64 
    7ff4:	41 54                	push   %r12
    7ff6:	55                   	push   %rbp
    7ff7:	48 63 ef             	movslq %edi,%rbp
    7ffa:	53                   	push   %rbx
    7ffb:	48 8d 1d de 92 00 00 	lea    0x92de(%rip),%rbx        # 112e0 <galois_mult_tables>
    8002:	4c 8b 24 eb          	mov    (%rbx,%rbp,8),%r12
    8006:	4d 85 e4             	test   %r12,%r12
    8009:	74 0d                	je     8018 <galois_get_mult_table+0x28>
    800b:	5b                   	pop    %rbx
    800c:	5d                   	pop    %rbp
    800d:	4c 89 e0             	mov    %r12,%rax
    8010:	41 5c                	pop    %r12
    8012:	c3                   	retq   
    8013:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8018:	e8 93 f5 ff ff       	callq  75b0 <galois_create_mult_tables>
    801d:	85 c0                	test   %eax,%eax
    801f:	75 ea                	jne    800b <galois_get_mult_table+0x1b>
    8021:	4c 8b 24 eb          	mov    (%rbx,%rbp,8),%r12
    8025:	5b                   	pop    %rbx
    8026:	5d                   	pop    %rbp
    8027:	4c 89 e0             	mov    %r12,%rax
    802a:	41 5c                	pop    %r12
    802c:	c3                   	retq   
    802d:	0f 1f 00             	nopl   (%rax)

0000000000008030 <galois_get_div_table>:
    8030:	f3 0f 1e fa          	endbr64 
    8034:	41 54                	push   %r12
    8036:	48 8d 05 a3 92 00 00 	lea    0x92a3(%rip),%rax        # 112e0 <galois_mult_tables>
    803d:	53                   	push   %rbx
    803e:	48 63 df             	movslq %edi,%rbx
    8041:	48 83 ec 08          	sub    $0x8,%rsp
    8045:	4c 8b 24 d8          	mov    (%rax,%rbx,8),%r12
    8049:	4d 85 e4             	test   %r12,%r12
    804c:	74 1a                	je     8068 <galois_get_div_table+0x38>
    804e:	48 8d 05 6b 91 00 00 	lea    0x916b(%rip),%rax        # 111c0 <galois_div_tables>
    8055:	4c 8b 24 d8          	mov    (%rax,%rbx,8),%r12
    8059:	48 83 c4 08          	add    $0x8,%rsp
    805d:	5b                   	pop    %rbx
    805e:	4c 89 e0             	mov    %r12,%rax
    8061:	41 5c                	pop    %r12
    8063:	c3                   	retq   
    8064:	0f 1f 40 00          	nopl   0x0(%rax)
    8068:	e8 43 f5 ff ff       	callq  75b0 <galois_create_mult_tables>
    806d:	85 c0                	test   %eax,%eax
    806f:	74 dd                	je     804e <galois_get_div_table+0x1e>
    8071:	eb e6                	jmp    8059 <galois_get_div_table+0x29>
    8073:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    807a:	00 00 00 00 
    807e:	66 90                	xchg   %ax,%ax

0000000000008080 <galois_get_log_table>:
    8080:	f3 0f 1e fa          	endbr64 
    8084:	41 54                	push   %r12
    8086:	55                   	push   %rbp
    8087:	48 63 ef             	movslq %edi,%rbp
    808a:	53                   	push   %rbx
    808b:	48 8d 1d 8e 94 00 00 	lea    0x948e(%rip),%rbx        # 11520 <galois_log_tables>
    8092:	4c 8b 24 eb          	mov    (%rbx,%rbp,8),%r12
    8096:	4d 85 e4             	test   %r12,%r12
    8099:	74 0d                	je     80a8 <galois_get_log_table+0x28>
    809b:	5b                   	pop    %rbx
    809c:	5d                   	pop    %rbp
    809d:	4c 89 e0             	mov    %r12,%rax
    80a0:	41 5c                	pop    %r12
    80a2:	c3                   	retq   
    80a3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    80a8:	83 ff 1e             	cmp    $0x1e,%edi
    80ab:	7f ee                	jg     809b <galois_get_log_table+0x1b>
    80ad:	e8 ae f2 ff ff       	callq  7360 <galois_create_log_tables.part.0>
    80b2:	85 c0                	test   %eax,%eax
    80b4:	75 e5                	jne    809b <galois_get_log_table+0x1b>
    80b6:	4c 8b 24 eb          	mov    (%rbx,%rbp,8),%r12
    80ba:	5b                   	pop    %rbx
    80bb:	5d                   	pop    %rbp
    80bc:	4c 89 e0             	mov    %r12,%rax
    80bf:	41 5c                	pop    %r12
    80c1:	c3                   	retq   
    80c2:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    80c9:	00 00 00 00 
    80cd:	0f 1f 00             	nopl   (%rax)

00000000000080d0 <galois_get_ilog_table>:
    80d0:	f3 0f 1e fa          	endbr64 
    80d4:	41 54                	push   %r12
    80d6:	55                   	push   %rbp
    80d7:	48 8d 2d 22 93 00 00 	lea    0x9322(%rip),%rbp        # 11400 <galois_ilog_tables>
    80de:	53                   	push   %rbx
    80df:	48 63 df             	movslq %edi,%rbx
    80e2:	4c 8b 64 dd 00       	mov    0x0(%rbp,%rbx,8),%r12
    80e7:	4d 85 e4             	test   %r12,%r12
    80ea:	74 0c                	je     80f8 <galois_get_ilog_table+0x28>
    80ec:	5b                   	pop    %rbx
    80ed:	5d                   	pop    %rbp
    80ee:	4c 89 e0             	mov    %r12,%rax
    80f1:	41 5c                	pop    %r12
    80f3:	c3                   	retq   
    80f4:	0f 1f 40 00          	nopl   0x0(%rax)
    80f8:	83 ff 1e             	cmp    $0x1e,%edi
    80fb:	7f ef                	jg     80ec <galois_get_ilog_table+0x1c>
    80fd:	48 8d 05 1c 94 00 00 	lea    0x941c(%rip),%rax        # 11520 <galois_log_tables>
    8104:	48 83 3c d8 00       	cmpq   $0x0,(%rax,%rbx,8)
    8109:	75 e1                	jne    80ec <galois_get_ilog_table+0x1c>
    810b:	e8 50 f2 ff ff       	callq  7360 <galois_create_log_tables.part.0>
    8110:	85 c0                	test   %eax,%eax
    8112:	75 d8                	jne    80ec <galois_get_ilog_table+0x1c>
    8114:	4c 8b 64 dd 00       	mov    0x0(%rbp,%rbx,8),%r12
    8119:	eb d1                	jmp    80ec <galois_get_ilog_table+0x1c>
    811b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000008120 <galois_region_xor>:
    8120:	f3 0f 1e fa          	endbr64 
    8124:	4c 63 c1             	movslq %ecx,%r8
    8127:	49 01 f8             	add    %rdi,%r8
    812a:	4c 39 c7             	cmp    %r8,%rdi
    812d:	73 28                	jae    8157 <galois_region_xor+0x37>
    812f:	48 89 f8             	mov    %rdi,%rax
    8132:	48 f7 d0             	not    %rax
    8135:	49 01 c0             	add    %rax,%r8
    8138:	49 c1 e8 03          	shr    $0x3,%r8
    813c:	31 c0                	xor    %eax,%eax
    813e:	66 90                	xchg   %ax,%ax
    8140:	48 8b 0c c7          	mov    (%rdi,%rax,8),%rcx
    8144:	48 33 0c c6          	xor    (%rsi,%rax,8),%rcx
    8148:	48 89 0c c2          	mov    %rcx,(%rdx,%rax,8)
    814c:	48 89 c1             	mov    %rax,%rcx
    814f:	48 ff c0             	inc    %rax
    8152:	49 39 c8             	cmp    %rcx,%r8
    8155:	75 e9                	jne    8140 <galois_region_xor+0x20>
    8157:	c3                   	retq   
    8158:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    815f:	00 

0000000000008160 <galois_create_split_w8_tables>:
    8160:	f3 0f 1e fa          	endbr64 
    8164:	48 83 3d 14 90 00 00 	cmpq   $0x0,0x9014(%rip)        # 11180 <galois_split_w8>
    816b:	00 
    816c:	74 03                	je     8171 <galois_create_split_w8_tables+0x11>
    816e:	31 c0                	xor    %eax,%eax
    8170:	c3                   	retq   
    8171:	41 57                	push   %r15
    8173:	bf 08 00 00 00       	mov    $0x8,%edi
    8178:	41 56                	push   %r14
    817a:	41 55                	push   %r13
    817c:	41 54                	push   %r12
    817e:	55                   	push   %rbp
    817f:	53                   	push   %rbx
    8180:	48 83 ec 18          	sub    $0x18,%rsp
    8184:	e8 27 f4 ff ff       	callq  75b0 <galois_create_mult_tables>
    8189:	85 c0                	test   %eax,%eax
    818b:	0f 88 00 01 00 00    	js     8291 <galois_create_split_w8_tables+0x131>
    8191:	31 db                	xor    %ebx,%ebx
    8193:	bf 00 00 04 00       	mov    $0x40000,%edi
    8198:	e8 63 92 ff ff       	callq  1400 <malloc@plt>
    819d:	48 8d 0d dc 8f 00 00 	lea    0x8fdc(%rip),%rcx        # 11180 <galois_split_w8>
    81a4:	48 89 04 d9          	mov    %rax,(%rcx,%rbx,8)
    81a8:	89 dd                	mov    %ebx,%ebp
    81aa:	48 85 c0             	test   %rax,%rax
    81ad:	0f 84 bc 00 00 00    	je     826f <galois_create_split_w8_tables+0x10f>
    81b3:	48 ff c3             	inc    %rbx
    81b6:	48 83 fb 07          	cmp    $0x7,%rbx
    81ba:	75 d7                	jne    8193 <galois_create_split_w8_tables+0x33>
    81bc:	c7 44 24 04 00 00 00 	movl   $0x0,0x4(%rsp)
    81c3:	00 
    81c4:	8b 44 24 04          	mov    0x4(%rsp),%eax
    81c8:	45 31 e4             	xor    %r12d,%r12d
    81cb:	85 c0                	test   %eax,%eax
    81cd:	41 0f 95 c4          	setne  %r12b
    81d1:	44 8d 34 c5 00 00 00 	lea    0x0(,%rax,8),%r14d
    81d8:	00 
    81d9:	44 01 e0             	add    %r12d,%eax
    81dc:	48 98                	cltq   
    81de:	48 8d 0d 9b 8f 00 00 	lea    0x8f9b(%rip),%rcx        # 11180 <galois_split_w8>
    81e5:	48 8d 04 c1          	lea    (%rcx,%rax,8),%rax
    81e9:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    81ee:	41 c1 e4 03          	shl    $0x3,%r12d
    81f2:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    81f7:	45 31 ed             	xor    %r13d,%r13d
    81fa:	48 8b 18             	mov    (%rax),%rbx
    81fd:	45 31 ff             	xor    %r15d,%r15d
    8200:	c4 c2 09 f7 ed       	shlx   %r14d,%r13d,%ebp
    8205:	0f 1f 00             	nopl   (%rax)
    8208:	ba 20 00 00 00       	mov    $0x20,%edx
    820d:	89 ef                	mov    %ebp,%edi
    820f:	c4 c2 19 f7 f7       	shlx   %r12d,%r15d,%esi
    8214:	e8 97 f6 ff ff       	callq  78b0 <galois_shift_multiply>
    8219:	42 89 04 bb          	mov    %eax,(%rbx,%r15,4)
    821d:	49 ff c7             	inc    %r15
    8220:	49 81 ff 00 01 00 00 	cmp    $0x100,%r15
    8227:	75 df                	jne    8208 <galois_create_split_w8_tables+0xa8>
    8229:	41 ff c5             	inc    %r13d
    822c:	48 81 c3 00 04 00 00 	add    $0x400,%rbx
    8233:	41 81 fd 00 01 00 00 	cmp    $0x100,%r13d
    823a:	75 c1                	jne    81fd <galois_create_split_w8_tables+0x9d>
    823c:	41 83 c4 08          	add    $0x8,%r12d
    8240:	48 83 44 24 08 08    	addq   $0x8,0x8(%rsp)
    8246:	41 83 fc 20          	cmp    $0x20,%r12d
    824a:	75 a6                	jne    81f2 <galois_create_split_w8_tables+0x92>
    824c:	83 44 24 04 03       	addl   $0x3,0x4(%rsp)
    8251:	8b 44 24 04          	mov    0x4(%rsp),%eax
    8255:	83 f8 06             	cmp    $0x6,%eax
    8258:	0f 85 66 ff ff ff    	jne    81c4 <galois_create_split_w8_tables+0x64>
    825e:	31 c0                	xor    %eax,%eax
    8260:	48 83 c4 18          	add    $0x18,%rsp
    8264:	5b                   	pop    %rbx
    8265:	5d                   	pop    %rbp
    8266:	41 5c                	pop    %r12
    8268:	41 5d                	pop    %r13
    826a:	41 5e                	pop    %r14
    826c:	41 5f                	pop    %r15
    826e:	c3                   	retq   
    826f:	8d 5b ff             	lea    -0x1(%rbx),%ebx
    8272:	85 ed                	test   %ebp,%ebp
    8274:	74 1b                	je     8291 <galois_create_split_w8_tables+0x131>
    8276:	48 63 db             	movslq %ebx,%rbx
    8279:	48 8d 05 00 8f 00 00 	lea    0x8f00(%rip),%rax        # 11180 <galois_split_w8>
    8280:	48 8b 3c d8          	mov    (%rax,%rbx,8),%rdi
    8284:	48 ff cb             	dec    %rbx
    8287:	e8 f4 8f ff ff       	callq  1280 <free@plt>
    828c:	83 fb ff             	cmp    $0xffffffff,%ebx
    828f:	75 e8                	jne    8279 <galois_create_split_w8_tables+0x119>
    8291:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    8296:	eb c8                	jmp    8260 <galois_create_split_w8_tables+0x100>
    8298:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    829f:	00 

00000000000082a0 <galois_w32_region_multiply>:
    82a0:	f3 0f 1e fa          	endbr64 
    82a4:	41 56                	push   %r14
    82a6:	48 63 d2             	movslq %edx,%rdx
    82a9:	41 55                	push   %r13
    82ab:	41 54                	push   %r12
    82ad:	55                   	push   %rbp
    82ae:	48 89 cd             	mov    %rcx,%rbp
    82b1:	53                   	push   %rbx
    82b2:	48 89 fb             	mov    %rdi,%rbx
    82b5:	48 83 ec 30          	sub    $0x30,%rsp
    82b9:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    82c0:	00 00 
    82c2:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    82c7:	31 c0                	xor    %eax,%eax
    82c9:	48 85 c9             	test   %rcx,%rcx
    82cc:	48 0f 44 ef          	cmove  %rdi,%rbp
    82d0:	48 c1 ea 02          	shr    $0x2,%rdx
    82d4:	48 83 3d a4 8e 00 00 	cmpq   $0x0,0x8ea4(%rip)        # 11180 <galois_split_w8>
    82db:	00 
    82dc:	49 89 d6             	mov    %rdx,%r14
    82df:	0f 84 28 01 00 00    	je     840d <galois_w32_region_multiply+0x16d>
    82e5:	48 8d 4c 24 10       	lea    0x10(%rsp),%rcx
    82ea:	48 89 cf             	mov    %rcx,%rdi
    82ed:	31 d2                	xor    %edx,%edx
    82ef:	c4 e2 6a f7 c6       	sarx   %edx,%esi,%eax
    82f4:	c1 e0 08             	shl    $0x8,%eax
    82f7:	25 ff ff 00 00       	and    $0xffff,%eax
    82fc:	83 c2 08             	add    $0x8,%edx
    82ff:	89 07                	mov    %eax,(%rdi)
    8301:	48 83 c7 04          	add    $0x4,%rdi
    8305:	83 fa 20             	cmp    $0x20,%edx
    8308:	75 e5                	jne    82ef <galois_w32_region_multiply+0x4f>
    830a:	45 85 c0             	test   %r8d,%r8d
    830d:	75 6b                	jne    837a <galois_w32_region_multiply+0xda>
    830f:	45 85 f6             	test   %r14d,%r14d
    8312:	0f 8e d8 00 00 00    	jle    83f0 <galois_w32_region_multiply+0x150>
    8318:	45 8d 6e ff          	lea    -0x1(%r14),%r13d
    831c:	45 31 e4             	xor    %r12d,%r12d
    831f:	90                   	nop
    8320:	46 8b 14 a3          	mov    (%rbx,%r12,4),%r10d
    8324:	4c 8d 1d 55 8e 00 00 	lea    0x8e55(%rip),%r11        # 11180 <galois_split_w8>
    832b:	31 ff                	xor    %edi,%edi
    832d:	45 31 c9             	xor    %r9d,%r9d
    8330:	46 8b 04 89          	mov    (%rcx,%r9,4),%r8d
    8334:	4c 89 de             	mov    %r11,%rsi
    8337:	31 d2                	xor    %edx,%edx
    8339:	c4 c2 6b f7 c2       	shrx   %edx,%r10d,%eax
    833e:	0f b6 c0             	movzbl %al,%eax
    8341:	4c 8b 36             	mov    (%rsi),%r14
    8344:	44 09 c0             	or     %r8d,%eax
    8347:	48 98                	cltq   
    8349:	83 c2 08             	add    $0x8,%edx
    834c:	41 33 3c 86          	xor    (%r14,%rax,4),%edi
    8350:	48 83 c6 08          	add    $0x8,%rsi
    8354:	83 fa 20             	cmp    $0x20,%edx
    8357:	75 e0                	jne    8339 <galois_w32_region_multiply+0x99>
    8359:	49 ff c1             	inc    %r9
    835c:	49 83 c3 08          	add    $0x8,%r11
    8360:	49 83 f9 04          	cmp    $0x4,%r9
    8364:	75 ca                	jne    8330 <galois_w32_region_multiply+0x90>
    8366:	42 89 7c a5 00       	mov    %edi,0x0(%rbp,%r12,4)
    836b:	49 8d 44 24 01       	lea    0x1(%r12),%rax
    8370:	4d 39 e5             	cmp    %r12,%r13
    8373:	74 7b                	je     83f0 <galois_w32_region_multiply+0x150>
    8375:	49 89 c4             	mov    %rax,%r12
    8378:	eb a6                	jmp    8320 <galois_w32_region_multiply+0x80>
    837a:	45 8d 6e ff          	lea    -0x1(%r14),%r13d
    837e:	45 31 e4             	xor    %r12d,%r12d
    8381:	45 85 f6             	test   %r14d,%r14d
    8384:	7e 6a                	jle    83f0 <galois_w32_region_multiply+0x150>
    8386:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    838d:	00 00 00 
    8390:	46 8b 14 a3          	mov    (%rbx,%r12,4),%r10d
    8394:	4c 8d 1d e5 8d 00 00 	lea    0x8de5(%rip),%r11        # 11180 <galois_split_w8>
    839b:	45 31 c9             	xor    %r9d,%r9d
    839e:	31 ff                	xor    %edi,%edi
    83a0:	46 8b 04 89          	mov    (%rcx,%r9,4),%r8d
    83a4:	4c 89 de             	mov    %r11,%rsi
    83a7:	31 d2                	xor    %edx,%edx
    83a9:	c4 c2 6b f7 c2       	shrx   %edx,%r10d,%eax
    83ae:	0f b6 c0             	movzbl %al,%eax
    83b1:	4c 8b 36             	mov    (%rsi),%r14
    83b4:	44 09 c0             	or     %r8d,%eax
    83b7:	48 98                	cltq   
    83b9:	83 c2 08             	add    $0x8,%edx
    83bc:	41 33 3c 86          	xor    (%r14,%rax,4),%edi
    83c0:	48 83 c6 08          	add    $0x8,%rsi
    83c4:	83 fa 20             	cmp    $0x20,%edx
    83c7:	75 e0                	jne    83a9 <galois_w32_region_multiply+0x109>
    83c9:	49 ff c1             	inc    %r9
    83cc:	49 83 c3 08          	add    $0x8,%r11
    83d0:	49 83 f9 04          	cmp    $0x4,%r9
    83d4:	75 ca                	jne    83a0 <galois_w32_region_multiply+0x100>
    83d6:	42 31 7c a5 00       	xor    %edi,0x0(%rbp,%r12,4)
    83db:	49 8d 44 24 01       	lea    0x1(%r12),%rax
    83e0:	4d 39 e5             	cmp    %r12,%r13
    83e3:	74 0b                	je     83f0 <galois_w32_region_multiply+0x150>
    83e5:	49 89 c4             	mov    %rax,%r12
    83e8:	eb a6                	jmp    8390 <galois_w32_region_multiply+0xf0>
    83ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    83f0:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    83f5:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    83fc:	00 00 
    83fe:	75 58                	jne    8458 <galois_w32_region_multiply+0x1b8>
    8400:	48 83 c4 30          	add    $0x30,%rsp
    8404:	5b                   	pop    %rbx
    8405:	5d                   	pop    %rbp
    8406:	41 5c                	pop    %r12
    8408:	41 5d                	pop    %r13
    840a:	41 5e                	pop    %r14
    840c:	c3                   	retq   
    840d:	bf 08 00 00 00       	mov    $0x8,%edi
    8412:	44 89 44 24 0c       	mov    %r8d,0xc(%rsp)
    8417:	89 74 24 08          	mov    %esi,0x8(%rsp)
    841b:	e8 40 fd ff ff       	callq  8160 <galois_create_split_w8_tables>
    8420:	85 c0                	test   %eax,%eax
    8422:	8b 74 24 08          	mov    0x8(%rsp),%esi
    8426:	44 8b 44 24 0c       	mov    0xc(%rsp),%r8d
    842b:	0f 89 b4 fe ff ff    	jns    82e5 <galois_w32_region_multiply+0x45>
    8431:	48 8b 0d 08 8d 00 00 	mov    0x8d08(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    8438:	48 8d 3d a9 25 00 00 	lea    0x25a9(%rip),%rdi        # a9e8 <__PRETTY_FUNCTION__.5741+0x30f>
    843f:	ba 47 00 00 00       	mov    $0x47,%edx
    8444:	be 01 00 00 00       	mov    $0x1,%esi
    8449:	e8 22 90 ff ff       	callq  1470 <fwrite@plt>
    844e:	bf 01 00 00 00       	mov    $0x1,%edi
    8453:	e8 08 90 ff ff       	callq  1460 <exit@plt>
    8458:	e8 b3 8e ff ff       	callq  1310 <__stack_chk_fail@plt>
    845d:	0f 1f 00             	nopl   (%rax)

0000000000008460 <galois_split_w8_multiply>:
    8460:	f3 0f 1e fa          	endbr64 
    8464:	53                   	push   %rbx
    8465:	4c 8d 1d 14 8d 00 00 	lea    0x8d14(%rip),%r11        # 11180 <galois_split_w8>
    846c:	45 31 c9             	xor    %r9d,%r9d
    846f:	45 31 d2             	xor    %r10d,%r10d
    8472:	46 8d 04 d5 00 00 00 	lea    0x0(,%r10,8),%r8d
    8479:	00 
    847a:	c4 62 3a f7 c7       	sarx   %r8d,%edi,%r8d
    847f:	41 c1 e0 08          	shl    $0x8,%r8d
    8483:	45 0f b7 c0          	movzwl %r8w,%r8d
    8487:	4c 89 d9             	mov    %r11,%rcx
    848a:	31 d2                	xor    %edx,%edx
    848c:	c4 e2 6a f7 c6       	sarx   %edx,%esi,%eax
    8491:	0f b6 c0             	movzbl %al,%eax
    8494:	48 8b 19             	mov    (%rcx),%rbx
    8497:	44 09 c0             	or     %r8d,%eax
    849a:	48 98                	cltq   
    849c:	83 c2 08             	add    $0x8,%edx
    849f:	44 33 0c 83          	xor    (%rbx,%rax,4),%r9d
    84a3:	48 83 c1 08          	add    $0x8,%rcx
    84a7:	83 fa 20             	cmp    $0x20,%edx
    84aa:	75 e0                	jne    848c <galois_split_w8_multiply+0x2c>
    84ac:	41 ff c2             	inc    %r10d
    84af:	49 83 c3 08          	add    $0x8,%r11
    84b3:	41 83 fa 04          	cmp    $0x4,%r10d
    84b7:	75 b9                	jne    8472 <galois_split_w8_multiply+0x12>
    84b9:	44 89 c8             	mov    %r9d,%eax
    84bc:	5b                   	pop    %rbx
    84bd:	c3                   	retq   
    84be:	66 90                	xchg   %ax,%ax

00000000000084c0 <galois_single_multiply.part.0>:
    84c0:	41 55                	push   %r13
    84c2:	48 8d 05 97 27 00 00 	lea    0x2797(%rip),%rax        # ac60 <mult_type>
    84c9:	41 54                	push   %r12
    84cb:	41 89 fc             	mov    %edi,%r12d
    84ce:	53                   	push   %rbx
    84cf:	48 63 da             	movslq %edx,%rbx
    84d2:	48 83 ec 10          	sub    $0x10,%rsp
    84d6:	8b 04 98             	mov    (%rax,%rbx,4),%eax
    84d9:	83 f8 0b             	cmp    $0xb,%eax
    84dc:	74 62                	je     8540 <galois_single_multiply.part.0+0x80>
    84de:	83 f8 0d             	cmp    $0xd,%eax
    84e1:	74 25                	je     8508 <galois_single_multiply.part.0+0x48>
    84e3:	83 f8 0e             	cmp    $0xe,%eax
    84e6:	0f 84 bc 00 00 00    	je     85a8 <galois_single_multiply.part.0+0xe8>
    84ec:	83 f8 0c             	cmp    $0xc,%eax
    84ef:	0f 85 47 01 00 00    	jne    863c <galois_single_multiply.part.0+0x17c>
    84f5:	48 83 c4 10          	add    $0x10,%rsp
    84f9:	5b                   	pop    %rbx
    84fa:	41 5c                	pop    %r12
    84fc:	41 5d                	pop    %r13
    84fe:	e9 ad f3 ff ff       	jmpq   78b0 <galois_shift_multiply>
    8503:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8508:	4c 8d 2d 11 90 00 00 	lea    0x9011(%rip),%r13        # 11520 <galois_log_tables>
    850f:	49 8b 4c dd 00       	mov    0x0(%r13,%rbx,8),%rcx
    8514:	48 85 c9             	test   %rcx,%rcx
    8517:	74 57                	je     8570 <galois_single_multiply.part.0+0xb0>
    8519:	48 63 f6             	movslq %esi,%rsi
    851c:	8b 04 b1             	mov    (%rcx,%rsi,4),%eax
    851f:	49 63 fc             	movslq %r12d,%rdi
    8522:	48 8d 15 d7 8e 00 00 	lea    0x8ed7(%rip),%rdx        # 11400 <galois_ilog_tables>
    8529:	48 8b 14 da          	mov    (%rdx,%rbx,8),%rdx
    852d:	03 04 b9             	add    (%rcx,%rdi,4),%eax
    8530:	48 98                	cltq   
    8532:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    8535:	48 83 c4 10          	add    $0x10,%rsp
    8539:	5b                   	pop    %rbx
    853a:	41 5c                	pop    %r12
    853c:	41 5d                	pop    %r13
    853e:	c3                   	retq   
    853f:	90                   	nop
    8540:	4c 8d 2d 99 8d 00 00 	lea    0x8d99(%rip),%r13        # 112e0 <galois_mult_tables>
    8547:	49 8b 44 dd 00       	mov    0x0(%r13,%rbx,8),%rax
    854c:	48 85 c0             	test   %rax,%rax
    854f:	74 77                	je     85c8 <galois_single_multiply.part.0+0x108>
    8551:	c4 42 69 f7 e4       	shlx   %edx,%r12d,%r12d
    8556:	41 09 f4             	or     %esi,%r12d
    8559:	4d 63 e4             	movslq %r12d,%r12
    855c:	42 8b 04 a0          	mov    (%rax,%r12,4),%eax
    8560:	48 83 c4 10          	add    $0x10,%rsp
    8564:	5b                   	pop    %rbx
    8565:	41 5c                	pop    %r12
    8567:	41 5d                	pop    %r13
    8569:	c3                   	retq   
    856a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    8570:	89 74 24 08          	mov    %esi,0x8(%rsp)
    8574:	83 fa 1e             	cmp    $0x1e,%edx
    8577:	0f 8f b4 00 00 00    	jg     8631 <galois_single_multiply.part.0+0x171>
    857d:	89 d7                	mov    %edx,%edi
    857f:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    8583:	e8 d8 ed ff ff       	callq  7360 <galois_create_log_tables.part.0>
    8588:	85 c0                	test   %eax,%eax
    858a:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    858e:	0f 88 9d 00 00 00    	js     8631 <galois_single_multiply.part.0+0x171>
    8594:	49 8b 4c dd 00       	mov    0x0(%r13,%rbx,8),%rcx
    8599:	8b 74 24 08          	mov    0x8(%rsp),%esi
    859d:	e9 77 ff ff ff       	jmpq   8519 <galois_single_multiply.part.0+0x59>
    85a2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    85a8:	48 83 3d d0 8b 00 00 	cmpq   $0x0,0x8bd0(%rip)        # 11180 <galois_split_w8>
    85af:	00 
    85b0:	74 3e                	je     85f0 <galois_single_multiply.part.0+0x130>
    85b2:	48 83 c4 10          	add    $0x10,%rsp
    85b6:	5b                   	pop    %rbx
    85b7:	44 89 e7             	mov    %r12d,%edi
    85ba:	41 5c                	pop    %r12
    85bc:	41 5d                	pop    %r13
    85be:	e9 9d fe ff ff       	jmpq   8460 <galois_split_w8_multiply>
    85c3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    85c8:	89 d7                	mov    %edx,%edi
    85ca:	89 54 24 08          	mov    %edx,0x8(%rsp)
    85ce:	89 74 24 0c          	mov    %esi,0xc(%rsp)
    85d2:	e8 d9 ef ff ff       	callq  75b0 <galois_create_mult_tables>
    85d7:	85 c0                	test   %eax,%eax
    85d9:	8b 54 24 08          	mov    0x8(%rsp),%edx
    85dd:	78 68                	js     8647 <galois_single_multiply.part.0+0x187>
    85df:	49 8b 44 dd 00       	mov    0x0(%r13,%rbx,8),%rax
    85e4:	8b 74 24 0c          	mov    0xc(%rsp),%esi
    85e8:	e9 64 ff ff ff       	jmpq   8551 <galois_single_multiply.part.0+0x91>
    85ed:	0f 1f 00             	nopl   (%rax)
    85f0:	31 c0                	xor    %eax,%eax
    85f2:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    85f6:	89 74 24 08          	mov    %esi,0x8(%rsp)
    85fa:	e8 61 fb ff ff       	callq  8160 <galois_create_split_w8_tables>
    85ff:	85 c0                	test   %eax,%eax
    8601:	8b 74 24 08          	mov    0x8(%rsp),%esi
    8605:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    8609:	79 a7                	jns    85b2 <galois_single_multiply.part.0+0xf2>
    860b:	89 d1                	mov    %edx,%ecx
    860d:	48 8d 15 84 24 00 00 	lea    0x2484(%rip),%rdx        # aa98 <__PRETTY_FUNCTION__.5741+0x3bf>
    8614:	48 8b 3d 25 8b 00 00 	mov    0x8b25(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    861b:	be 01 00 00 00       	mov    $0x1,%esi
    8620:	31 c0                	xor    %eax,%eax
    8622:	e8 59 8e ff ff       	callq  1480 <__fprintf_chk@plt>
    8627:	bf 01 00 00 00       	mov    $0x1,%edi
    862c:	e8 2f 8e ff ff       	callq  1460 <exit@plt>
    8631:	89 d1                	mov    %edx,%ecx
    8633:	48 8d 15 2e 24 00 00 	lea    0x242e(%rip),%rdx        # aa68 <__PRETTY_FUNCTION__.5741+0x38f>
    863a:	eb d8                	jmp    8614 <galois_single_multiply.part.0+0x154>
    863c:	89 d1                	mov    %edx,%ecx
    863e:	48 8d 15 8b 24 00 00 	lea    0x248b(%rip),%rdx        # aad0 <__PRETTY_FUNCTION__.5741+0x3f7>
    8645:	eb cd                	jmp    8614 <galois_single_multiply.part.0+0x154>
    8647:	89 d1                	mov    %edx,%ecx
    8649:	48 8d 15 e0 23 00 00 	lea    0x23e0(%rip),%rdx        # aa30 <__PRETTY_FUNCTION__.5741+0x357>
    8650:	eb c2                	jmp    8614 <galois_single_multiply.part.0+0x154>
    8652:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    8659:	00 00 00 00 
    865d:	0f 1f 00             	nopl   (%rax)

0000000000008660 <galois_single_multiply>:
    8660:	f3 0f 1e fa          	endbr64 
    8664:	85 ff                	test   %edi,%edi
    8666:	74 10                	je     8678 <galois_single_multiply+0x18>
    8668:	85 f6                	test   %esi,%esi
    866a:	74 0c                	je     8678 <galois_single_multiply+0x18>
    866c:	e9 4f fe ff ff       	jmpq   84c0 <galois_single_multiply.part.0>
    8671:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8678:	31 c0                	xor    %eax,%eax
    867a:	c3                   	retq   
    867b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000008680 <galois_single_divide>:
    8680:	f3 0f 1e fa          	endbr64 
    8684:	41 56                	push   %r14
    8686:	48 8d 05 d3 25 00 00 	lea    0x25d3(%rip),%rax        # ac60 <mult_type>
    868d:	41 55                	push   %r13
    868f:	41 89 fd             	mov    %edi,%r13d
    8692:	41 54                	push   %r12
    8694:	55                   	push   %rbp
    8695:	48 63 ee             	movslq %esi,%rbp
    8698:	53                   	push   %rbx
    8699:	48 63 da             	movslq %edx,%rbx
    869c:	48 83 ec 10          	sub    $0x10,%rsp
    86a0:	8b 04 98             	mov    (%rax,%rbx,4),%eax
    86a3:	83 f8 0b             	cmp    $0xb,%eax
    86a6:	74 78                	je     8720 <galois_single_divide+0xa0>
    86a8:	83 f8 0d             	cmp    $0xd,%eax
    86ab:	74 23                	je     86d0 <galois_single_divide+0x50>
    86ad:	85 ed                	test   %ebp,%ebp
    86af:	0f 84 0c 01 00 00    	je     87c1 <galois_single_divide+0x141>
    86b5:	85 ff                	test   %edi,%edi
    86b7:	0f 85 93 00 00 00    	jne    8750 <galois_single_divide+0xd0>
    86bd:	31 c0                	xor    %eax,%eax
    86bf:	48 83 c4 10          	add    $0x10,%rsp
    86c3:	5b                   	pop    %rbx
    86c4:	5d                   	pop    %rbp
    86c5:	41 5c                	pop    %r12
    86c7:	41 5d                	pop    %r13
    86c9:	41 5e                	pop    %r14
    86cb:	c3                   	retq   
    86cc:	0f 1f 40 00          	nopl   0x0(%rax)
    86d0:	85 ed                	test   %ebp,%ebp
    86d2:	0f 84 f3 00 00 00    	je     87cb <galois_single_divide+0x14b>
    86d8:	85 ff                	test   %edi,%edi
    86da:	74 e1                	je     86bd <galois_single_divide+0x3d>
    86dc:	4c 8d 35 3d 8e 00 00 	lea    0x8e3d(%rip),%r14        # 11520 <galois_log_tables>
    86e3:	49 8b 0c de          	mov    (%r14,%rbx,8),%rcx
    86e7:	48 85 c9             	test   %rcx,%rcx
    86ea:	0f 84 b0 00 00 00    	je     87a0 <galois_single_divide+0x120>
    86f0:	4d 63 e5             	movslq %r13d,%r12
    86f3:	42 8b 04 a1          	mov    (%rcx,%r12,4),%eax
    86f7:	48 8d 15 02 8d 00 00 	lea    0x8d02(%rip),%rdx        # 11400 <galois_ilog_tables>
    86fe:	48 8b 14 da          	mov    (%rdx,%rbx,8),%rdx
    8702:	2b 04 a9             	sub    (%rcx,%rbp,4),%eax
    8705:	48 98                	cltq   
    8707:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    870a:	48 83 c4 10          	add    $0x10,%rsp
    870e:	5b                   	pop    %rbx
    870f:	5d                   	pop    %rbp
    8710:	41 5c                	pop    %r12
    8712:	41 5d                	pop    %r13
    8714:	41 5e                	pop    %r14
    8716:	c3                   	retq   
    8717:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    871e:	00 00 
    8720:	4c 8d 35 99 8a 00 00 	lea    0x8a99(%rip),%r14        # 111c0 <galois_div_tables>
    8727:	49 8b 04 de          	mov    (%r14,%rbx,8),%rax
    872b:	48 85 c0             	test   %rax,%rax
    872e:	74 50                	je     8780 <galois_single_divide+0x100>
    8730:	c4 42 69 f7 e5       	shlx   %edx,%r13d,%r12d
    8735:	41 09 ec             	or     %ebp,%r12d
    8738:	4d 63 e4             	movslq %r12d,%r12
    873b:	42 8b 04 a0          	mov    (%rax,%r12,4),%eax
    873f:	48 83 c4 10          	add    $0x10,%rsp
    8743:	5b                   	pop    %rbx
    8744:	5d                   	pop    %rbp
    8745:	41 5c                	pop    %r12
    8747:	41 5d                	pop    %r13
    8749:	41 5e                	pop    %r14
    874b:	c3                   	retq   
    874c:	0f 1f 40 00          	nopl   0x0(%rax)
    8750:	89 d6                	mov    %edx,%esi
    8752:	89 ef                	mov    %ebp,%edi
    8754:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    8758:	e8 b3 00 00 00       	callq  8810 <galois_inverse>
    875d:	89 c6                	mov    %eax,%esi
    875f:	85 c0                	test   %eax,%eax
    8761:	0f 84 56 ff ff ff    	je     86bd <galois_single_divide+0x3d>
    8767:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    876b:	48 83 c4 10          	add    $0x10,%rsp
    876f:	5b                   	pop    %rbx
    8770:	5d                   	pop    %rbp
    8771:	41 5c                	pop    %r12
    8773:	44 89 ef             	mov    %r13d,%edi
    8776:	41 5d                	pop    %r13
    8778:	41 5e                	pop    %r14
    877a:	e9 41 fd ff ff       	jmpq   84c0 <galois_single_multiply.part.0>
    877f:	90                   	nop
    8780:	89 d7                	mov    %edx,%edi
    8782:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    8786:	e8 25 ee ff ff       	callq  75b0 <galois_create_mult_tables>
    878b:	85 c0                	test   %eax,%eax
    878d:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    8791:	78 68                	js     87fb <galois_single_divide+0x17b>
    8793:	49 8b 04 de          	mov    (%r14,%rbx,8),%rax
    8797:	eb 97                	jmp    8730 <galois_single_divide+0xb0>
    8799:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    87a0:	83 fa 1e             	cmp    $0x1e,%edx
    87a3:	7f 30                	jg     87d5 <galois_single_divide+0x155>
    87a5:	89 d7                	mov    %edx,%edi
    87a7:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    87ab:	e8 b0 eb ff ff       	callq  7360 <galois_create_log_tables.part.0>
    87b0:	85 c0                	test   %eax,%eax
    87b2:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    87b6:	78 1d                	js     87d5 <galois_single_divide+0x155>
    87b8:	49 8b 0c de          	mov    (%r14,%rbx,8),%rcx
    87bc:	e9 2f ff ff ff       	jmpq   86f0 <galois_single_divide+0x70>
    87c1:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    87c6:	e9 f4 fe ff ff       	jmpq   86bf <galois_single_divide+0x3f>
    87cb:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    87d0:	e9 ea fe ff ff       	jmpq   86bf <galois_single_divide+0x3f>
    87d5:	89 d1                	mov    %edx,%ecx
    87d7:	48 8d 15 8a 22 00 00 	lea    0x228a(%rip),%rdx        # aa68 <__PRETTY_FUNCTION__.5741+0x38f>
    87de:	48 8b 3d 5b 89 00 00 	mov    0x895b(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    87e5:	be 01 00 00 00       	mov    $0x1,%esi
    87ea:	31 c0                	xor    %eax,%eax
    87ec:	e8 8f 8c ff ff       	callq  1480 <__fprintf_chk@plt>
    87f1:	bf 01 00 00 00       	mov    $0x1,%edi
    87f6:	e8 65 8c ff ff       	callq  1460 <exit@plt>
    87fb:	89 d1                	mov    %edx,%ecx
    87fd:	48 8d 15 2c 22 00 00 	lea    0x222c(%rip),%rdx        # aa30 <__PRETTY_FUNCTION__.5741+0x357>
    8804:	eb d8                	jmp    87de <galois_single_divide+0x15e>
    8806:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    880d:	00 00 00 

0000000000008810 <galois_inverse>:
    8810:	f3 0f 1e fa          	endbr64 
    8814:	89 f2                	mov    %esi,%edx
    8816:	85 ff                	test   %edi,%edi
    8818:	74 2e                	je     8848 <galois_inverse+0x38>
    881a:	48 63 c6             	movslq %esi,%rax
    881d:	48 8d 0d 3c 24 00 00 	lea    0x243c(%rip),%rcx        # ac60 <mult_type>
    8824:	8b 04 81             	mov    (%rcx,%rax,4),%eax
    8827:	83 e0 fd             	and    $0xfffffffd,%eax
    882a:	83 f8 0c             	cmp    $0xc,%eax
    882d:	74 11                	je     8840 <galois_inverse+0x30>
    882f:	89 fe                	mov    %edi,%esi
    8831:	bf 01 00 00 00       	mov    $0x1,%edi
    8836:	e9 45 fe ff ff       	jmpq   8680 <galois_single_divide>
    883b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8840:	e9 8b f6 ff ff       	jmpq   7ed0 <galois_shift_inverse>
    8845:	0f 1f 00             	nopl   (%rax)
    8848:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    884d:	c3                   	retq   
    884e:	66 90                	xchg   %ax,%ax

0000000000008850 <reed_sol_r6_coding_matrix>:
    8850:	f3 0f 1e fa          	endbr64 
    8854:	41 55                	push   %r13
    8856:	8d 46 f8             	lea    -0x8(%rsi),%eax
    8859:	4c 63 ef             	movslq %edi,%r13
    885c:	41 54                	push   %r12
    885e:	55                   	push   %rbp
    885f:	89 f5                	mov    %esi,%ebp
    8861:	53                   	push   %rbx
    8862:	48 83 ec 08          	sub    $0x8,%rsp
    8866:	83 e0 f7             	and    $0xfffffff7,%eax
    8869:	74 09                	je     8874 <reed_sol_r6_coding_matrix+0x24>
    886b:	83 fe 20             	cmp    $0x20,%esi
    886e:	0f 85 8c 00 00 00    	jne    8900 <reed_sol_r6_coding_matrix+0xb0>
    8874:	43 8d 7c 2d 00       	lea    0x0(%r13,%r13,1),%edi
    8879:	48 63 ff             	movslq %edi,%rdi
    887c:	48 c1 e7 02          	shl    $0x2,%rdi
    8880:	e8 7b 8b ff ff       	callq  1400 <malloc@plt>
    8885:	49 89 c4             	mov    %rax,%r12
    8888:	48 85 c0             	test   %rax,%rax
    888b:	74 73                	je     8900 <reed_sol_r6_coding_matrix+0xb0>
    888d:	45 85 ed             	test   %r13d,%r13d
    8890:	0f 8e 82 00 00 00    	jle    8918 <reed_sol_r6_coding_matrix+0xc8>
    8896:	41 8d 55 ff          	lea    -0x1(%r13),%edx
    889a:	48 8d 54 90 04       	lea    0x4(%rax,%rdx,4),%rdx
    889f:	90                   	nop
    88a0:	c7 00 01 00 00 00    	movl   $0x1,(%rax)
    88a6:	48 83 c0 04          	add    $0x4,%rax
    88aa:	48 39 d0             	cmp    %rdx,%rax
    88ad:	75 f1                	jne    88a0 <reed_sol_r6_coding_matrix+0x50>
    88af:	49 63 c5             	movslq %r13d,%rax
    88b2:	49 8d 1c 84          	lea    (%r12,%rax,4),%rbx
    88b6:	c7 03 01 00 00 00    	movl   $0x1,(%rbx)
    88bc:	41 83 fd 01          	cmp    $0x1,%r13d
    88c0:	7e 30                	jle    88f2 <reed_sol_r6_coding_matrix+0xa2>
    88c2:	41 8d 55 fe          	lea    -0x2(%r13),%edx
    88c6:	48 8d 44 10 01       	lea    0x1(%rax,%rdx,1),%rax
    88cb:	4d 8d 2c 84          	lea    (%r12,%rax,4),%r13
    88cf:	bf 01 00 00 00       	mov    $0x1,%edi
    88d4:	0f 1f 40 00          	nopl   0x0(%rax)
    88d8:	89 ea                	mov    %ebp,%edx
    88da:	be 02 00 00 00       	mov    $0x2,%esi
    88df:	e8 7c fd ff ff       	callq  8660 <galois_single_multiply>
    88e4:	89 43 04             	mov    %eax,0x4(%rbx)
    88e7:	48 83 c3 04          	add    $0x4,%rbx
    88eb:	89 c7                	mov    %eax,%edi
    88ed:	4c 39 eb             	cmp    %r13,%rbx
    88f0:	75 e6                	jne    88d8 <reed_sol_r6_coding_matrix+0x88>
    88f2:	48 83 c4 08          	add    $0x8,%rsp
    88f6:	5b                   	pop    %rbx
    88f7:	5d                   	pop    %rbp
    88f8:	4c 89 e0             	mov    %r12,%rax
    88fb:	41 5c                	pop    %r12
    88fd:	41 5d                	pop    %r13
    88ff:	c3                   	retq   
    8900:	48 83 c4 08          	add    $0x8,%rsp
    8904:	5b                   	pop    %rbx
    8905:	45 31 e4             	xor    %r12d,%r12d
    8908:	5d                   	pop    %rbp
    8909:	4c 89 e0             	mov    %r12,%rax
    890c:	41 5c                	pop    %r12
    890e:	41 5d                	pop    %r13
    8910:	c3                   	retq   
    8911:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8918:	42 c7 04 a8 01 00 00 	movl   $0x1,(%rax,%r13,4)
    891f:	00 
    8920:	eb d0                	jmp    88f2 <reed_sol_r6_coding_matrix+0xa2>
    8922:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    8929:	00 00 00 00 
    892d:	0f 1f 00             	nopl   (%rax)

0000000000008930 <reed_sol_galois_w32_region_multby_2>:
    8930:	f3 0f 1e fa          	endbr64 
    8934:	55                   	push   %rbp
    8935:	89 f5                	mov    %esi,%ebp
    8937:	53                   	push   %rbx
    8938:	48 89 fb             	mov    %rdi,%rbx
    893b:	48 83 ec 08          	sub    $0x8,%rsp
    893f:	83 3d 42 57 00 00 ff 	cmpl   $0xffffffff,0x5742(%rip)        # e088 <prim32>
    8946:	74 40                	je     8988 <reed_sol_galois_w32_region_multby_2+0x58>
    8948:	48 63 f5             	movslq %ebp,%rsi
    894b:	48 01 de             	add    %rbx,%rsi
    894e:	8b 3d 34 57 00 00    	mov    0x5734(%rip),%edi        # e088 <prim32>
    8954:	48 39 f3             	cmp    %rsi,%rbx
    8957:	73 21                	jae    897a <reed_sol_galois_w32_region_multby_2+0x4a>
    8959:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8960:	8b 13                	mov    (%rbx),%edx
    8962:	8d 04 12             	lea    (%rdx,%rdx,1),%eax
    8965:	89 c1                	mov    %eax,%ecx
    8967:	31 f9                	xor    %edi,%ecx
    8969:	85 d2                	test   %edx,%edx
    896b:	0f 48 c1             	cmovs  %ecx,%eax
    896e:	48 83 c3 04          	add    $0x4,%rbx
    8972:	89 43 fc             	mov    %eax,-0x4(%rbx)
    8975:	48 39 de             	cmp    %rbx,%rsi
    8978:	77 e6                	ja     8960 <reed_sol_galois_w32_region_multby_2+0x30>
    897a:	48 83 c4 08          	add    $0x8,%rsp
    897e:	5b                   	pop    %rbx
    897f:	5d                   	pop    %rbp
    8980:	c3                   	retq   
    8981:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8988:	ba 20 00 00 00       	mov    $0x20,%edx
    898d:	be 02 00 00 00       	mov    $0x2,%esi
    8992:	bf 00 00 00 80       	mov    $0x80000000,%edi
    8997:	e8 c4 fc ff ff       	callq  8660 <galois_single_multiply>
    899c:	89 05 e6 56 00 00    	mov    %eax,0x56e6(%rip)        # e088 <prim32>
    89a2:	eb a4                	jmp    8948 <reed_sol_galois_w32_region_multby_2+0x18>
    89a4:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    89ab:	00 00 00 00 
    89af:	90                   	nop

00000000000089b0 <reed_sol_galois_w08_region_multby_2>:
    89b0:	f3 0f 1e fa          	endbr64 
    89b4:	55                   	push   %rbp
    89b5:	89 f5                	mov    %esi,%ebp
    89b7:	53                   	push   %rbx
    89b8:	48 89 fb             	mov    %rdi,%rbx
    89bb:	48 83 ec 08          	sub    $0x8,%rsp
    89bf:	83 3d be 56 00 00 ff 	cmpl   $0xffffffff,0x56be(%rip)        # e084 <prim08>
    89c6:	74 58                	je     8a20 <reed_sol_galois_w08_region_multby_2+0x70>
    89c8:	48 63 f5             	movslq %ebp,%rsi
    89cb:	48 01 de             	add    %rbx,%rsi
    89ce:	48 39 f3             	cmp    %rsi,%rbx
    89d1:	73 41                	jae    8a14 <reed_sol_galois_w08_region_multby_2+0x64>
    89d3:	44 8b 0d a6 56 00 00 	mov    0x56a6(%rip),%r9d        # e080 <mask08_1>
    89da:	44 8b 05 9b 56 00 00 	mov    0x569b(%rip),%r8d        # e07c <mask08_2>
    89e1:	8b 3d 9d 56 00 00    	mov    0x569d(%rip),%edi        # e084 <prim08>
    89e7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    89ee:	00 00 
    89f0:	8b 13                	mov    (%rbx),%edx
    89f2:	48 83 c3 04          	add    $0x4,%rbx
    89f6:	89 d1                	mov    %edx,%ecx
    89f8:	44 21 c1             	and    %r8d,%ecx
    89fb:	8d 04 09             	lea    (%rcx,%rcx,1),%eax
    89fe:	c1 e9 07             	shr    $0x7,%ecx
    8a01:	29 c8                	sub    %ecx,%eax
    8a03:	01 d2                	add    %edx,%edx
    8a05:	21 f8                	and    %edi,%eax
    8a07:	44 21 ca             	and    %r9d,%edx
    8a0a:	31 d0                	xor    %edx,%eax
    8a0c:	89 43 fc             	mov    %eax,-0x4(%rbx)
    8a0f:	48 39 de             	cmp    %rbx,%rsi
    8a12:	77 dc                	ja     89f0 <reed_sol_galois_w08_region_multby_2+0x40>
    8a14:	48 83 c4 08          	add    $0x8,%rsp
    8a18:	5b                   	pop    %rbx
    8a19:	5d                   	pop    %rbp
    8a1a:	c3                   	retq   
    8a1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8a20:	ba 08 00 00 00       	mov    $0x8,%edx
    8a25:	be 02 00 00 00       	mov    $0x2,%esi
    8a2a:	bf 80 00 00 00       	mov    $0x80,%edi
    8a2f:	e8 2c fc ff ff       	callq  8660 <galois_single_multiply>
    8a34:	c7 05 46 56 00 00 00 	movl   $0x0,0x5646(%rip)        # e084 <prim08>
    8a3b:	00 00 00 
    8a3e:	85 c0                	test   %eax,%eax
    8a40:	74 13                	je     8a55 <reed_sol_galois_w08_region_multby_2+0xa5>
    8a42:	31 d2                	xor    %edx,%edx
    8a44:	0f 1f 40 00          	nopl   0x0(%rax)
    8a48:	09 c2                	or     %eax,%edx
    8a4a:	c1 e0 08             	shl    $0x8,%eax
    8a4d:	75 f9                	jne    8a48 <reed_sol_galois_w08_region_multby_2+0x98>
    8a4f:	89 15 2f 56 00 00    	mov    %edx,0x562f(%rip)        # e084 <prim08>
    8a55:	c7 05 21 56 00 00 fe 	movl   $0xfefefefe,0x5621(%rip)        # e080 <mask08_1>
    8a5c:	fe fe fe 
    8a5f:	c7 05 13 56 00 00 80 	movl   $0x80808080,0x5613(%rip)        # e07c <mask08_2>
    8a66:	80 80 80 
    8a69:	e9 5a ff ff ff       	jmpq   89c8 <reed_sol_galois_w08_region_multby_2+0x18>
    8a6e:	66 90                	xchg   %ax,%ax

0000000000008a70 <reed_sol_galois_w16_region_multby_2>:
    8a70:	f3 0f 1e fa          	endbr64 
    8a74:	55                   	push   %rbp
    8a75:	89 f5                	mov    %esi,%ebp
    8a77:	53                   	push   %rbx
    8a78:	48 89 fb             	mov    %rdi,%rbx
    8a7b:	48 83 ec 08          	sub    $0x8,%rsp
    8a7f:	83 3d f2 55 00 00 ff 	cmpl   $0xffffffff,0x55f2(%rip)        # e078 <prim16>
    8a86:	74 58                	je     8ae0 <reed_sol_galois_w16_region_multby_2+0x70>
    8a88:	48 63 f5             	movslq %ebp,%rsi
    8a8b:	48 01 de             	add    %rbx,%rsi
    8a8e:	48 39 f3             	cmp    %rsi,%rbx
    8a91:	73 41                	jae    8ad4 <reed_sol_galois_w16_region_multby_2+0x64>
    8a93:	44 8b 0d da 55 00 00 	mov    0x55da(%rip),%r9d        # e074 <mask16_1>
    8a9a:	44 8b 05 cf 55 00 00 	mov    0x55cf(%rip),%r8d        # e070 <mask16_2>
    8aa1:	8b 3d d1 55 00 00    	mov    0x55d1(%rip),%edi        # e078 <prim16>
    8aa7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    8aae:	00 00 
    8ab0:	8b 13                	mov    (%rbx),%edx
    8ab2:	48 83 c3 04          	add    $0x4,%rbx
    8ab6:	89 d1                	mov    %edx,%ecx
    8ab8:	44 21 c1             	and    %r8d,%ecx
    8abb:	8d 04 09             	lea    (%rcx,%rcx,1),%eax
    8abe:	c1 e9 0f             	shr    $0xf,%ecx
    8ac1:	29 c8                	sub    %ecx,%eax
    8ac3:	01 d2                	add    %edx,%edx
    8ac5:	21 f8                	and    %edi,%eax
    8ac7:	44 21 ca             	and    %r9d,%edx
    8aca:	31 d0                	xor    %edx,%eax
    8acc:	89 43 fc             	mov    %eax,-0x4(%rbx)
    8acf:	48 39 de             	cmp    %rbx,%rsi
    8ad2:	77 dc                	ja     8ab0 <reed_sol_galois_w16_region_multby_2+0x40>
    8ad4:	48 83 c4 08          	add    $0x8,%rsp
    8ad8:	5b                   	pop    %rbx
    8ad9:	5d                   	pop    %rbp
    8ada:	c3                   	retq   
    8adb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8ae0:	ba 10 00 00 00       	mov    $0x10,%edx
    8ae5:	be 02 00 00 00       	mov    $0x2,%esi
    8aea:	bf 00 80 00 00       	mov    $0x8000,%edi
    8aef:	e8 6c fb ff ff       	callq  8660 <galois_single_multiply>
    8af4:	c7 05 7a 55 00 00 00 	movl   $0x0,0x557a(%rip)        # e078 <prim16>
    8afb:	00 00 00 
    8afe:	85 c0                	test   %eax,%eax
    8b00:	74 13                	je     8b15 <reed_sol_galois_w16_region_multby_2+0xa5>
    8b02:	31 d2                	xor    %edx,%edx
    8b04:	0f 1f 40 00          	nopl   0x0(%rax)
    8b08:	09 c2                	or     %eax,%edx
    8b0a:	c1 e0 10             	shl    $0x10,%eax
    8b0d:	75 f9                	jne    8b08 <reed_sol_galois_w16_region_multby_2+0x98>
    8b0f:	89 15 63 55 00 00    	mov    %edx,0x5563(%rip)        # e078 <prim16>
    8b15:	c7 05 55 55 00 00 fe 	movl   $0xfffefffe,0x5555(%rip)        # e074 <mask16_1>
    8b1c:	ff fe ff 
    8b1f:	c7 05 47 55 00 00 00 	movl   $0x80008000,0x5547(%rip)        # e070 <mask16_2>
    8b26:	80 00 80 
    8b29:	e9 5a ff ff ff       	jmpq   8a88 <reed_sol_galois_w16_region_multby_2+0x18>
    8b2e:	66 90                	xchg   %ax,%ax

0000000000008b30 <reed_sol_r6_encode>:
    8b30:	f3 0f 1e fa          	endbr64 
    8b34:	41 57                	push   %r15
    8b36:	49 63 c0             	movslq %r8d,%rax
    8b39:	49 89 cf             	mov    %rcx,%r15
    8b3c:	41 56                	push   %r14
    8b3e:	41 89 fe             	mov    %edi,%r14d
    8b41:	41 55                	push   %r13
    8b43:	49 89 d5             	mov    %rdx,%r13
    8b46:	41 54                	push   %r12
    8b48:	41 89 f4             	mov    %esi,%r12d
    8b4b:	55                   	push   %rbp
    8b4c:	53                   	push   %rbx
    8b4d:	48 89 c3             	mov    %rax,%rbx
    8b50:	48 83 ec 28          	sub    $0x28,%rsp
    8b54:	48 8b 32             	mov    (%rdx),%rsi
    8b57:	48 8b 39             	mov    (%rcx),%rdi
    8b5a:	48 89 c2             	mov    %rax,%rdx
    8b5d:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    8b62:	e8 59 88 ff ff       	callq  13c0 <memcpy@plt>
    8b67:	49 63 c6             	movslq %r14d,%rax
    8b6a:	49 8d 44 c5 f8       	lea    -0x8(%r13,%rax,8),%rax
    8b6f:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    8b74:	41 8d 46 fe          	lea    -0x2(%r14),%eax
    8b78:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    8b7c:	41 83 fe 01          	cmp    $0x1,%r14d
    8b80:	0f 8e ca 00 00 00    	jle    8c50 <reed_sol_r6_encode+0x120>
    8b86:	89 c2                	mov    %eax,%edx
    8b88:	49 8d 6d 08          	lea    0x8(%r13),%rbp
    8b8c:	4d 8d 74 d5 10       	lea    0x10(%r13,%rdx,8),%r14
    8b91:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8b98:	49 8b 3f             	mov    (%r15),%rdi
    8b9b:	48 8b 75 00          	mov    0x0(%rbp),%rsi
    8b9f:	89 d9                	mov    %ebx,%ecx
    8ba1:	48 89 fa             	mov    %rdi,%rdx
    8ba4:	48 83 c5 08          	add    $0x8,%rbp
    8ba8:	e8 73 f5 ff ff       	callq  8120 <galois_region_xor>
    8bad:	49 39 ee             	cmp    %rbp,%r14
    8bb0:	75 e6                	jne    8b98 <reed_sol_r6_encode+0x68>
    8bb2:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    8bb7:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8bbb:	48 8b 30             	mov    (%rax),%rsi
    8bbe:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    8bc3:	e8 f8 87 ff ff       	callq  13c0 <memcpy@plt>
    8bc8:	48 63 6c 24 0c       	movslq 0xc(%rsp),%rbp
    8bcd:	41 83 fc 10          	cmp    $0x10,%r12d
    8bd1:	74 1d                	je     8bf0 <reed_sol_r6_encode+0xc0>
    8bd3:	41 83 fc 20          	cmp    $0x20,%r12d
    8bd7:	74 67                	je     8c40 <reed_sol_r6_encode+0x110>
    8bd9:	41 83 fc 08          	cmp    $0x8,%r12d
    8bdd:	74 51                	je     8c30 <reed_sol_r6_encode+0x100>
    8bdf:	48 83 c4 28          	add    $0x28,%rsp
    8be3:	5b                   	pop    %rbx
    8be4:	5d                   	pop    %rbp
    8be5:	41 5c                	pop    %r12
    8be7:	41 5d                	pop    %r13
    8be9:	41 5e                	pop    %r14
    8beb:	31 c0                	xor    %eax,%eax
    8bed:	41 5f                	pop    %r15
    8bef:	c3                   	retq   
    8bf0:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8bf4:	89 de                	mov    %ebx,%esi
    8bf6:	e8 75 fe ff ff       	callq  8a70 <reed_sol_galois_w16_region_multby_2>
    8bfb:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8bff:	49 8b 74 ed 00       	mov    0x0(%r13,%rbp,8),%rsi
    8c04:	89 d9                	mov    %ebx,%ecx
    8c06:	48 89 fa             	mov    %rdi,%rdx
    8c09:	48 ff cd             	dec    %rbp
    8c0c:	e8 0f f5 ff ff       	callq  8120 <galois_region_xor>
    8c11:	85 ed                	test   %ebp,%ebp
    8c13:	79 b8                	jns    8bcd <reed_sol_r6_encode+0x9d>
    8c15:	48 83 c4 28          	add    $0x28,%rsp
    8c19:	5b                   	pop    %rbx
    8c1a:	5d                   	pop    %rbp
    8c1b:	41 5c                	pop    %r12
    8c1d:	41 5d                	pop    %r13
    8c1f:	41 5e                	pop    %r14
    8c21:	b8 01 00 00 00       	mov    $0x1,%eax
    8c26:	41 5f                	pop    %r15
    8c28:	c3                   	retq   
    8c29:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8c30:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8c34:	89 de                	mov    %ebx,%esi
    8c36:	e8 75 fd ff ff       	callq  89b0 <reed_sol_galois_w08_region_multby_2>
    8c3b:	eb be                	jmp    8bfb <reed_sol_r6_encode+0xcb>
    8c3d:	0f 1f 00             	nopl   (%rax)
    8c40:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8c44:	89 de                	mov    %ebx,%esi
    8c46:	e8 e5 fc ff ff       	callq  8930 <reed_sol_galois_w32_region_multby_2>
    8c4b:	eb ae                	jmp    8bfb <reed_sol_r6_encode+0xcb>
    8c4d:	0f 1f 00             	nopl   (%rax)
    8c50:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    8c55:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8c59:	48 8b 30             	mov    (%rax),%rsi
    8c5c:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    8c61:	e8 5a 87 ff ff       	callq  13c0 <memcpy@plt>
    8c66:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    8c6a:	85 c0                	test   %eax,%eax
    8c6c:	0f 89 56 ff ff ff    	jns    8bc8 <reed_sol_r6_encode+0x98>
    8c72:	eb a1                	jmp    8c15 <reed_sol_r6_encode+0xe5>
    8c74:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    8c7b:	00 00 00 00 
    8c7f:	90                   	nop

0000000000008c80 <reed_sol_extended_vandermonde_matrix>:
    8c80:	f3 0f 1e fa          	endbr64 
    8c84:	41 57                	push   %r15
    8c86:	41 56                	push   %r14
    8c88:	41 55                	push   %r13
    8c8a:	41 54                	push   %r12
    8c8c:	41 89 f4             	mov    %esi,%r12d
    8c8f:	55                   	push   %rbp
    8c90:	89 fd                	mov    %edi,%ebp
    8c92:	53                   	push   %rbx
    8c93:	89 d3                	mov    %edx,%ebx
    8c95:	48 83 ec 18          	sub    $0x18,%rsp
    8c99:	83 fa 1d             	cmp    $0x1d,%edx
    8c9c:	7f 1a                	jg     8cb8 <reed_sol_extended_vandermonde_matrix+0x38>
    8c9e:	b8 01 00 00 00       	mov    $0x1,%eax
    8ca3:	c4 e2 69 f7 c0       	shlx   %edx,%eax,%eax
    8ca8:	39 f8                	cmp    %edi,%eax
    8caa:	0f 8c 19 01 00 00    	jl     8dc9 <reed_sol_extended_vandermonde_matrix+0x149>
    8cb0:	39 f0                	cmp    %esi,%eax
    8cb2:	0f 8c 11 01 00 00    	jl     8dc9 <reed_sol_extended_vandermonde_matrix+0x149>
    8cb8:	41 89 ee             	mov    %ebp,%r14d
    8cbb:	45 0f af f4          	imul   %r12d,%r14d
    8cbf:	49 63 fe             	movslq %r14d,%rdi
    8cc2:	48 c1 e7 02          	shl    $0x2,%rdi
    8cc6:	e8 35 87 ff ff       	callq  1400 <malloc@plt>
    8ccb:	49 89 c5             	mov    %rax,%r13
    8cce:	48 85 c0             	test   %rax,%rax
    8cd1:	0f 84 f2 00 00 00    	je     8dc9 <reed_sol_extended_vandermonde_matrix+0x149>
    8cd7:	c7 00 01 00 00 00    	movl   $0x1,(%rax)
    8cdd:	41 83 fc 01          	cmp    $0x1,%r12d
    8ce1:	0f 8e e7 00 00 00    	jle    8dce <reed_sol_extended_vandermonde_matrix+0x14e>
    8ce7:	41 8d 54 24 fe       	lea    -0x2(%r12),%edx
    8cec:	48 8d 40 04          	lea    0x4(%rax),%rax
    8cf0:	49 8d 54 95 08       	lea    0x8(%r13,%rdx,4),%rdx
    8cf5:	0f 1f 00             	nopl   (%rax)
    8cf8:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
    8cfe:	48 83 c0 04          	add    $0x4,%rax
    8d02:	48 39 d0             	cmp    %rdx,%rax
    8d05:	75 f1                	jne    8cf8 <reed_sol_extended_vandermonde_matrix+0x78>
    8d07:	83 fd 01             	cmp    $0x1,%ebp
    8d0a:	0f 84 a7 00 00 00    	je     8db7 <reed_sol_extended_vandermonde_matrix+0x137>
    8d10:	8d 45 ff             	lea    -0x1(%rbp),%eax
    8d13:	89 44 24 08          	mov    %eax,0x8(%rsp)
    8d17:	44 89 f0             	mov    %r14d,%eax
    8d1a:	44 29 e0             	sub    %r12d,%eax
    8d1d:	48 98                	cltq   
    8d1f:	49 8d 54 85 00       	lea    0x0(%r13,%rax,4),%rdx
    8d24:	41 8d 4c 24 ff       	lea    -0x1(%r12),%ecx
    8d29:	31 c0                	xor    %eax,%eax
    8d2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8d30:	c7 04 82 00 00 00 00 	movl   $0x0,(%rdx,%rax,4)
    8d37:	48 ff c0             	inc    %rax
    8d3a:	39 c1                	cmp    %eax,%ecx
    8d3c:	7f f2                	jg     8d30 <reed_sol_extended_vandermonde_matrix+0xb0>
    8d3e:	41 ff ce             	dec    %r14d
    8d41:	4d 63 f6             	movslq %r14d,%r14
    8d44:	43 c7 44 b5 00 01 00 	movl   $0x1,0x0(%r13,%r14,4)
    8d4b:	00 00 
    8d4d:	83 fd 02             	cmp    $0x2,%ebp
    8d50:	74 65                	je     8db7 <reed_sol_extended_vandermonde_matrix+0x137>
    8d52:	83 7c 24 08 01       	cmpl   $0x1,0x8(%rsp)
    8d57:	7e 5e                	jle    8db7 <reed_sol_extended_vandermonde_matrix+0x137>
    8d59:	41 8d 44 24 ff       	lea    -0x1(%r12),%eax
    8d5e:	44 89 64 24 04       	mov    %r12d,0x4(%rsp)
    8d63:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    8d67:	41 bf 01 00 00 00    	mov    $0x1,%r15d
    8d6d:	0f 1f 00             	nopl   (%rax)
    8d70:	45 85 e4             	test   %r12d,%r12d
    8d73:	7e 33                	jle    8da8 <reed_sol_extended_vandermonde_matrix+0x128>
    8d75:	48 63 54 24 04       	movslq 0x4(%rsp),%rdx
    8d7a:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    8d7e:	4d 8d 74 95 00       	lea    0x0(%r13,%rdx,4),%r14
    8d83:	48 01 d0             	add    %rdx,%rax
    8d86:	49 8d 6c 85 04       	lea    0x4(%r13,%rax,4),%rbp
    8d8b:	bf 01 00 00 00       	mov    $0x1,%edi
    8d90:	41 89 3e             	mov    %edi,(%r14)
    8d93:	89 da                	mov    %ebx,%edx
    8d95:	44 89 fe             	mov    %r15d,%esi
    8d98:	e8 c3 f8 ff ff       	callq  8660 <galois_single_multiply>
    8d9d:	49 83 c6 04          	add    $0x4,%r14
    8da1:	89 c7                	mov    %eax,%edi
    8da3:	4c 39 f5             	cmp    %r14,%rbp
    8da6:	75 e8                	jne    8d90 <reed_sol_extended_vandermonde_matrix+0x110>
    8da8:	41 ff c7             	inc    %r15d
    8dab:	44 01 64 24 04       	add    %r12d,0x4(%rsp)
    8db0:	44 39 7c 24 08       	cmp    %r15d,0x8(%rsp)
    8db5:	75 b9                	jne    8d70 <reed_sol_extended_vandermonde_matrix+0xf0>
    8db7:	48 83 c4 18          	add    $0x18,%rsp
    8dbb:	5b                   	pop    %rbx
    8dbc:	5d                   	pop    %rbp
    8dbd:	41 5c                	pop    %r12
    8dbf:	4c 89 e8             	mov    %r13,%rax
    8dc2:	41 5d                	pop    %r13
    8dc4:	41 5e                	pop    %r14
    8dc6:	41 5f                	pop    %r15
    8dc8:	c3                   	retq   
    8dc9:	45 31 ed             	xor    %r13d,%r13d
    8dcc:	eb e9                	jmp    8db7 <reed_sol_extended_vandermonde_matrix+0x137>
    8dce:	83 fd 01             	cmp    $0x1,%ebp
    8dd1:	74 e4                	je     8db7 <reed_sol_extended_vandermonde_matrix+0x137>
    8dd3:	8d 45 ff             	lea    -0x1(%rbp),%eax
    8dd6:	89 44 24 08          	mov    %eax,0x8(%rsp)
    8dda:	45 29 e6             	sub    %r12d,%r14d
    8ddd:	e9 5f ff ff ff       	jmpq   8d41 <reed_sol_extended_vandermonde_matrix+0xc1>
    8de2:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    8de9:	00 00 00 00 
    8ded:	0f 1f 00             	nopl   (%rax)

0000000000008df0 <reed_sol_big_vandermonde_distribution_matrix>:
    8df0:	f3 0f 1e fa          	endbr64 
    8df4:	41 57                	push   %r15
    8df6:	41 56                	push   %r14
    8df8:	41 55                	push   %r13
    8dfa:	41 54                	push   %r12
    8dfc:	55                   	push   %rbp
    8dfd:	53                   	push   %rbx
    8dfe:	48 83 ec 78          	sub    $0x78,%rsp
    8e02:	89 7c 24 04          	mov    %edi,0x4(%rsp)
    8e06:	89 74 24 6c          	mov    %esi,0x6c(%rsp)
    8e0a:	39 fe                	cmp    %edi,%esi
    8e0c:	0f 8d ed 03 00 00    	jge    91ff <reed_sol_big_vandermonde_distribution_matrix+0x40f>
    8e12:	89 f3                	mov    %esi,%ebx
    8e14:	41 89 d6             	mov    %edx,%r14d
    8e17:	e8 64 fe ff ff       	callq  8c80 <reed_sol_extended_vandermonde_matrix>
    8e1c:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    8e21:	48 89 c1             	mov    %rax,%rcx
    8e24:	48 85 c0             	test   %rax,%rax
    8e27:	0f 84 d2 03 00 00    	je     91ff <reed_sol_big_vandermonde_distribution_matrix+0x40f>
    8e2d:	83 fb 01             	cmp    $0x1,%ebx
    8e30:	0f 8e ca 01 00 00    	jle    9000 <reed_sol_big_vandermonde_distribution_matrix+0x210>
    8e36:	48 63 fb             	movslq %ebx,%rdi
    8e39:	8d 43 fe             	lea    -0x2(%rbx),%eax
    8e3c:	48 8d 14 bd 00 00 00 	lea    0x0(,%rdi,4),%rdx
    8e43:	00 
    8e44:	48 83 c0 02          	add    $0x2,%rax
    8e48:	48 01 d1             	add    %rdx,%rcx
    8e4b:	48 89 54 24 08       	mov    %rdx,0x8(%rsp)
    8e50:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    8e55:	48 8d 57 01          	lea    0x1(%rdi),%rdx
    8e59:	8d 43 ff             	lea    -0x1(%rbx),%eax
    8e5c:	48 89 7c 24 48       	mov    %rdi,0x48(%rsp)
    8e61:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    8e66:	48 89 54 24 58       	mov    %rdx,0x58(%rsp)
    8e6b:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
    8e70:	48 89 7c 24 50       	mov    %rdi,0x50(%rsp)
    8e75:	48 c7 44 24 20 01 00 	movq   $0x1,0x20(%rsp)
    8e7c:	00 00 
    8e7e:	89 44 24 68          	mov    %eax,0x68(%rsp)
    8e82:	44 89 74 24 44       	mov    %r14d,0x44(%rsp)
    8e87:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    8e8e:	00 00 
    8e90:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    8e95:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    8e9a:	89 fe                	mov    %edi,%esi
    8e9c:	89 fa                	mov    %edi,%edx
    8e9e:	4c 8b 44 24 48       	mov    0x48(%rsp),%r8
    8ea3:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    8ea8:	44 8b 4c 24 04       	mov    0x4(%rsp),%r9d
    8ead:	0f 1f 00             	nopl   (%rax)
    8eb0:	44 8b 14 87          	mov    (%rdi,%rax,4),%r10d
    8eb4:	89 c1                	mov    %eax,%ecx
    8eb6:	45 85 d2             	test   %r10d,%r10d
    8eb9:	75 45                	jne    8f00 <reed_sol_big_vandermonde_distribution_matrix+0x110>
    8ebb:	ff c2                	inc    %edx
    8ebd:	4c 01 c0             	add    %r8,%rax
    8ec0:	41 39 d1             	cmp    %edx,%r9d
    8ec3:	7f eb                	jg     8eb0 <reed_sol_big_vandermonde_distribution_matrix+0xc0>
    8ec5:	44 8b 74 24 44       	mov    0x44(%rsp),%r14d
    8eca:	48 8b 3d 6f 82 00 00 	mov    0x826f(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    8ed1:	44 8b 44 24 6c       	mov    0x6c(%rsp),%r8d
    8ed6:	8b 4c 24 04          	mov    0x4(%rsp),%ecx
    8eda:	45 89 f1             	mov    %r14d,%r9d
    8edd:	48 8d 15 a4 1e 00 00 	lea    0x1ea4(%rip),%rdx        # ad88 <prim_poly+0x88>
    8ee4:	be 01 00 00 00       	mov    $0x1,%esi
    8ee9:	31 c0                	xor    %eax,%eax
    8eeb:	e8 90 85 ff ff       	callq  1480 <__fprintf_chk@plt>
    8ef0:	bf 01 00 00 00       	mov    $0x1,%edi
    8ef5:	e8 66 85 ff ff       	callq  1460 <exit@plt>
    8efa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    8f00:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    8f05:	89 44 24 40          	mov    %eax,0x40(%rsp)
    8f09:	39 c2                	cmp    %eax,%edx
    8f0b:	0f 85 4f 02 00 00    	jne    9160 <reed_sol_big_vandermonde_distribution_matrix+0x370>
    8f11:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    8f16:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    8f1b:	8b 34 88             	mov    (%rax,%rcx,4),%esi
    8f1e:	83 fe 01             	cmp    $0x1,%esi
    8f21:	0f 85 79 02 00 00    	jne    91a0 <reed_sol_big_vandermonde_distribution_matrix+0x3b0>
    8f27:	8b 44 24 68          	mov    0x68(%rsp),%eax
    8f2b:	45 31 ff             	xor    %r15d,%r15d
    8f2e:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    8f33:	eb 15                	jmp    8f4a <reed_sol_big_vandermonde_distribution_matrix+0x15a>
    8f35:	0f 1f 00             	nopl   (%rax)
    8f38:	49 8d 47 01          	lea    0x1(%r15),%rax
    8f3c:	4c 3b 7c 24 18       	cmp    0x18(%rsp),%r15
    8f41:	0f 84 81 00 00 00    	je     8fc8 <reed_sol_big_vandermonde_distribution_matrix+0x1d8>
    8f47:	49 89 c7             	mov    %rax,%r15
    8f4a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    8f4f:	46 8b 2c b8          	mov    (%rax,%r15,4),%r13d
    8f53:	44 39 7c 24 40       	cmp    %r15d,0x40(%rsp)
    8f58:	74 de                	je     8f38 <reed_sol_big_vandermonde_distribution_matrix+0x148>
    8f5a:	45 85 ed             	test   %r13d,%r13d
    8f5d:	74 d9                	je     8f38 <reed_sol_big_vandermonde_distribution_matrix+0x148>
    8f5f:	8b 54 24 04          	mov    0x4(%rsp),%edx
    8f63:	85 d2                	test   %edx,%edx
    8f65:	7e d1                	jle    8f38 <reed_sol_big_vandermonde_distribution_matrix+0x148>
    8f67:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    8f6c:	4c 8b 74 24 20       	mov    0x20(%rsp),%r14
    8f71:	4a 8d 2c b8          	lea    (%rax,%r15,4),%rbp
    8f75:	4d 29 fe             	sub    %r15,%r14
    8f78:	45 31 e4             	xor    %r12d,%r12d
    8f7b:	4c 89 7c 24 30       	mov    %r15,0x30(%rsp)
    8f80:	49 89 ef             	mov    %rbp,%r15
    8f83:	44 89 e5             	mov    %r12d,%ebp
    8f86:	4d 89 f4             	mov    %r14,%r12
    8f89:	44 8b 74 24 44       	mov    0x44(%rsp),%r14d
    8f8e:	66 90                	xchg   %ax,%ax
    8f90:	43 8b 34 a7          	mov    (%r15,%r12,4),%esi
    8f94:	41 8b 1f             	mov    (%r15),%ebx
    8f97:	44 89 f2             	mov    %r14d,%edx
    8f9a:	44 89 ef             	mov    %r13d,%edi
    8f9d:	e8 be f6 ff ff       	callq  8660 <galois_single_multiply>
    8fa2:	31 c3                	xor    %eax,%ebx
    8fa4:	ff c5                	inc    %ebp
    8fa6:	41 89 1f             	mov    %ebx,(%r15)
    8fa9:	4c 03 7c 24 08       	add    0x8(%rsp),%r15
    8fae:	39 6c 24 04          	cmp    %ebp,0x4(%rsp)
    8fb2:	75 dc                	jne    8f90 <reed_sol_big_vandermonde_distribution_matrix+0x1a0>
    8fb4:	4c 8b 7c 24 30       	mov    0x30(%rsp),%r15
    8fb9:	49 8d 47 01          	lea    0x1(%r15),%rax
    8fbd:	4c 3b 7c 24 18       	cmp    0x18(%rsp),%r15
    8fc2:	0f 85 7f ff ff ff    	jne    8f47 <reed_sol_big_vandermonde_distribution_matrix+0x157>
    8fc8:	48 ff 44 24 20       	incq   0x20(%rsp)
    8fcd:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    8fd2:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    8fd7:	48 01 7c 24 10       	add    %rdi,0x10(%rsp)
    8fdc:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    8fe1:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    8fe6:	48 01 4c 24 50       	add    %rcx,0x50(%rsp)
    8feb:	48 01 7c 24 38       	add    %rdi,0x38(%rsp)
    8ff0:	48 3b 44 24 60       	cmp    0x60(%rsp),%rax
    8ff5:	0f 85 95 fe ff ff    	jne    8e90 <reed_sol_big_vandermonde_distribution_matrix+0xa0>
    8ffb:	44 8b 74 24 44       	mov    0x44(%rsp),%r14d
    9000:	8b 4c 24 6c          	mov    0x6c(%rsp),%ecx
    9004:	89 c8                	mov    %ecx,%eax
    9006:	0f af c1             	imul   %ecx,%eax
    9009:	85 c9                	test   %ecx,%ecx
    900b:	0f 8e 9d 00 00 00    	jle    90ae <reed_sol_big_vandermonde_distribution_matrix+0x2be>
    9011:	48 63 d8             	movslq %eax,%rbx
    9014:	48 63 c1             	movslq %ecx,%rax
    9017:	44 8d 68 ff          	lea    -0x1(%rax),%r13d
    901b:	48 8d 6b 01          	lea    0x1(%rbx),%rbp
    901f:	4d 8d 64 2d 00       	lea    0x0(%r13,%rbp,1),%r12
    9024:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    9029:	4c 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%r15
    9030:	00 
    9031:	48 89 d9             	mov    %rbx,%rcx
    9034:	eb 17                	jmp    904d <reed_sol_big_vandermonde_distribution_matrix+0x25d>
    9036:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    903d:	00 00 00 
    9040:	48 89 e9             	mov    %rbp,%rcx
    9043:	48 39 6c 24 08       	cmp    %rbp,0x8(%rsp)
    9048:	74 64                	je     90ae <reed_sol_big_vandermonde_distribution_matrix+0x2be>
    904a:	48 ff c5             	inc    %rbp
    904d:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9052:	8b 34 88             	mov    (%rax,%rcx,4),%esi
    9055:	83 fe 01             	cmp    $0x1,%esi
    9058:	74 e6                	je     9040 <reed_sol_big_vandermonde_distribution_matrix+0x250>
    905a:	44 89 f2             	mov    %r14d,%edx
    905d:	bf 01 00 00 00       	mov    $0x1,%edi
    9062:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    9067:	e8 14 f6 ff ff       	callq  8680 <galois_single_divide>
    906c:	41 89 c5             	mov    %eax,%r13d
    906f:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    9074:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9079:	44 8b 64 24 6c       	mov    0x6c(%rsp),%r12d
    907e:	48 8d 1c 88          	lea    (%rax,%rcx,4),%rbx
    9082:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    9088:	8b 33                	mov    (%rbx),%esi
    908a:	44 89 f2             	mov    %r14d,%edx
    908d:	44 89 ef             	mov    %r13d,%edi
    9090:	e8 cb f5 ff ff       	callq  8660 <galois_single_multiply>
    9095:	41 ff c4             	inc    %r12d
    9098:	89 03                	mov    %eax,(%rbx)
    909a:	4c 01 fb             	add    %r15,%rbx
    909d:	44 39 64 24 04       	cmp    %r12d,0x4(%rsp)
    90a2:	75 e4                	jne    9088 <reed_sol_big_vandermonde_distribution_matrix+0x298>
    90a4:	48 89 e9             	mov    %rbp,%rcx
    90a7:	48 39 6c 24 08       	cmp    %rbp,0x8(%rsp)
    90ac:	75 9c                	jne    904a <reed_sol_big_vandermonde_distribution_matrix+0x25a>
    90ae:	8b 44 24 6c          	mov    0x6c(%rsp),%eax
    90b2:	8d 48 01             	lea    0x1(%rax),%ecx
    90b5:	89 cd                	mov    %ecx,%ebp
    90b7:	0f af e8             	imul   %eax,%ebp
    90ba:	3b 4c 24 04          	cmp    0x4(%rsp),%ecx
    90be:	0f 8d 44 01 00 00    	jge    9208 <reed_sol_big_vandermonde_distribution_matrix+0x418>
    90c4:	4c 63 f8             	movslq %eax,%r15
    90c7:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    90cc:	48 63 ed             	movslq %ebp,%rbp
    90cf:	ff c8                	dec    %eax
    90d1:	4e 8d 24 bd 00 00 00 	lea    0x0(,%r15,4),%r12
    90d8:	00 
    90d9:	48 8d 44 05 01       	lea    0x1(%rbp,%rax,1),%rax
    90de:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    90e3:	4c 89 7c 24 10       	mov    %r15,0x10(%rsp)
    90e8:	48 8d 1c 87          	lea    (%rdi,%rax,4),%rbx
    90ec:	41 89 cc             	mov    %ecx,%r12d
    90ef:	eb 1f                	jmp    9110 <reed_sol_big_vandermonde_distribution_matrix+0x320>
    90f1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    90f8:	41 ff c4             	inc    %r12d
    90fb:	48 03 6c 24 10       	add    0x10(%rsp),%rbp
    9100:	48 03 5c 24 08       	add    0x8(%rsp),%rbx
    9105:	44 39 64 24 04       	cmp    %r12d,0x4(%rsp)
    910a:	0f 84 f8 00 00 00    	je     9208 <reed_sol_big_vandermonde_distribution_matrix+0x418>
    9110:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9115:	8b 34 a8             	mov    (%rax,%rbp,4),%esi
    9118:	83 fe 01             	cmp    $0x1,%esi
    911b:	74 db                	je     90f8 <reed_sol_big_vandermonde_distribution_matrix+0x308>
    911d:	44 89 f2             	mov    %r14d,%edx
    9120:	bf 01 00 00 00       	mov    $0x1,%edi
    9125:	e8 56 f5 ff ff       	callq  8680 <galois_single_divide>
    912a:	41 89 c5             	mov    %eax,%r13d
    912d:	8b 44 24 6c          	mov    0x6c(%rsp),%eax
    9131:	85 c0                	test   %eax,%eax
    9133:	7e c3                	jle    90f8 <reed_sol_big_vandermonde_distribution_matrix+0x308>
    9135:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    913a:	4c 8d 3c a8          	lea    (%rax,%rbp,4),%r15
    913e:	66 90                	xchg   %ax,%ax
    9140:	41 8b 3f             	mov    (%r15),%edi
    9143:	44 89 f2             	mov    %r14d,%edx
    9146:	44 89 ee             	mov    %r13d,%esi
    9149:	e8 12 f5 ff ff       	callq  8660 <galois_single_multiply>
    914e:	41 89 07             	mov    %eax,(%r15)
    9151:	49 83 c7 04          	add    $0x4,%r15
    9155:	49 39 df             	cmp    %rbx,%r15
    9158:	75 e6                	jne    9140 <reed_sol_big_vandermonde_distribution_matrix+0x350>
    915a:	eb 9c                	jmp    90f8 <reed_sol_big_vandermonde_distribution_matrix+0x308>
    915c:	0f 1f 40 00          	nopl   0x0(%rax)
    9160:	8b 54 24 68          	mov    0x68(%rsp),%edx
    9164:	29 f1                	sub    %esi,%ecx
    9166:	48 63 c9             	movslq %ecx,%rcx
    9169:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    916e:	48 01 ca             	add    %rcx,%rdx
    9171:	48 8d 04 8f          	lea    (%rdi,%rcx,4),%rax
    9175:	48 8d 7c 97 04       	lea    0x4(%rdi,%rdx,4),%rdi
    917a:	48 8b 54 24 50       	mov    0x50(%rsp),%rdx
    917f:	48 29 ca             	sub    %rcx,%rdx
    9182:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    9188:	8b 08                	mov    (%rax),%ecx
    918a:	8b 34 90             	mov    (%rax,%rdx,4),%esi
    918d:	89 30                	mov    %esi,(%rax)
    918f:	89 0c 90             	mov    %ecx,(%rax,%rdx,4)
    9192:	48 83 c0 04          	add    $0x4,%rax
    9196:	48 39 f8             	cmp    %rdi,%rax
    9199:	75 ed                	jne    9188 <reed_sol_big_vandermonde_distribution_matrix+0x398>
    919b:	e9 71 fd ff ff       	jmpq   8f11 <reed_sol_big_vandermonde_distribution_matrix+0x121>
    91a0:	44 8b 74 24 44       	mov    0x44(%rsp),%r14d
    91a5:	bf 01 00 00 00       	mov    $0x1,%edi
    91aa:	44 89 f2             	mov    %r14d,%edx
    91ad:	e8 ce f4 ff ff       	callq  8680 <galois_single_divide>
    91b2:	44 8b 6c 24 04       	mov    0x4(%rsp),%r13d
    91b7:	89 c5                	mov    %eax,%ebp
    91b9:	45 85 ed             	test   %r13d,%r13d
    91bc:	0f 8e 65 fd ff ff    	jle    8f27 <reed_sol_big_vandermonde_distribution_matrix+0x137>
    91c2:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    91c7:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    91cc:	4c 8b 64 24 08       	mov    0x8(%rsp),%r12
    91d1:	4c 8d 3c 88          	lea    (%rax,%rcx,4),%r15
    91d5:	31 db                	xor    %ebx,%ebx
    91d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    91de:	00 00 
    91e0:	41 8b 37             	mov    (%r15),%esi
    91e3:	44 89 f2             	mov    %r14d,%edx
    91e6:	89 ef                	mov    %ebp,%edi
    91e8:	e8 73 f4 ff ff       	callq  8660 <galois_single_multiply>
    91ed:	ff c3                	inc    %ebx
    91ef:	41 89 07             	mov    %eax,(%r15)
    91f2:	4d 01 e7             	add    %r12,%r15
    91f5:	41 39 dd             	cmp    %ebx,%r13d
    91f8:	75 e6                	jne    91e0 <reed_sol_big_vandermonde_distribution_matrix+0x3f0>
    91fa:	e9 28 fd ff ff       	jmpq   8f27 <reed_sol_big_vandermonde_distribution_matrix+0x137>
    91ff:	48 c7 44 24 28 00 00 	movq   $0x0,0x28(%rsp)
    9206:	00 00 
    9208:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    920d:	48 83 c4 78          	add    $0x78,%rsp
    9211:	5b                   	pop    %rbx
    9212:	5d                   	pop    %rbp
    9213:	41 5c                	pop    %r12
    9215:	41 5d                	pop    %r13
    9217:	41 5e                	pop    %r14
    9219:	41 5f                	pop    %r15
    921b:	c3                   	retq   
    921c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000009220 <reed_sol_vandermonde_coding_matrix>:
    9220:	f3 0f 1e fa          	endbr64 
    9224:	41 55                	push   %r13
    9226:	41 54                	push   %r12
    9228:	55                   	push   %rbp
    9229:	89 f5                	mov    %esi,%ebp
    922b:	53                   	push   %rbx
    922c:	89 fb                	mov    %edi,%ebx
    922e:	01 f7                	add    %esi,%edi
    9230:	48 83 ec 08          	sub    $0x8,%rsp
    9234:	89 de                	mov    %ebx,%esi
    9236:	e8 b5 fb ff ff       	callq  8df0 <reed_sol_big_vandermonde_distribution_matrix>
    923b:	48 85 c0             	test   %rax,%rax
    923e:	74 60                	je     92a0 <reed_sol_vandermonde_coding_matrix+0x80>
    9240:	0f af eb             	imul   %ebx,%ebp
    9243:	49 89 c5             	mov    %rax,%r13
    9246:	48 63 fd             	movslq %ebp,%rdi
    9249:	48 c1 e7 02          	shl    $0x2,%rdi
    924d:	e8 ae 81 ff ff       	callq  1400 <malloc@plt>
    9252:	49 89 c4             	mov    %rax,%r12
    9255:	48 85 c0             	test   %rax,%rax
    9258:	74 28                	je     9282 <reed_sol_vandermonde_coding_matrix+0x62>
    925a:	0f af db             	imul   %ebx,%ebx
    925d:	85 ed                	test   %ebp,%ebp
    925f:	7e 21                	jle    9282 <reed_sol_vandermonde_coding_matrix+0x62>
    9261:	48 63 db             	movslq %ebx,%rbx
    9264:	8d 75 ff             	lea    -0x1(%rbp),%esi
    9267:	49 8d 44 9d 00       	lea    0x0(%r13,%rbx,4),%rax
    926c:	31 d2                	xor    %edx,%edx
    926e:	66 90                	xchg   %ax,%ax
    9270:	8b 0c 90             	mov    (%rax,%rdx,4),%ecx
    9273:	41 89 0c 94          	mov    %ecx,(%r12,%rdx,4)
    9277:	48 89 d1             	mov    %rdx,%rcx
    927a:	48 ff c2             	inc    %rdx
    927d:	48 39 ce             	cmp    %rcx,%rsi
    9280:	75 ee                	jne    9270 <reed_sol_vandermonde_coding_matrix+0x50>
    9282:	4c 89 ef             	mov    %r13,%rdi
    9285:	e8 f6 7f ff ff       	callq  1280 <free@plt>
    928a:	48 83 c4 08          	add    $0x8,%rsp
    928e:	5b                   	pop    %rbx
    928f:	5d                   	pop    %rbp
    9290:	4c 89 e0             	mov    %r12,%rax
    9293:	41 5c                	pop    %r12
    9295:	41 5d                	pop    %r13
    9297:	c3                   	retq   
    9298:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    929f:	00 
    92a0:	48 83 c4 08          	add    $0x8,%rsp
    92a4:	5b                   	pop    %rbx
    92a5:	45 31 e4             	xor    %r12d,%r12d
    92a8:	5d                   	pop    %rbp
    92a9:	4c 89 e0             	mov    %r12,%rax
    92ac:	41 5c                	pop    %r12
    92ae:	41 5d                	pop    %r13
    92b0:	c3                   	retq   
    92b1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    92b8:	00 00 00 
    92bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000092c0 <cauchy_n_ones>:
    92c0:	f3 0f 1e fa          	endbr64 
    92c4:	41 57                	push   %r15
    92c6:	41 56                	push   %r14
    92c8:	4c 8d 35 d1 7d 00 00 	lea    0x7dd1(%rip),%r14        # 110a0 <PPs>
    92cf:	41 55                	push   %r13
    92d1:	4c 63 ee             	movslq %esi,%r13
    92d4:	41 8d 45 ff          	lea    -0x1(%r13),%eax
    92d8:	41 54                	push   %r12
    92da:	41 bc 01 00 00 00    	mov    $0x1,%r12d
    92e0:	c4 42 79 f7 e4       	shlx   %eax,%r12d,%r12d
    92e5:	55                   	push   %rbp
    92e6:	4c 89 ed             	mov    %r13,%rbp
    92e9:	53                   	push   %rbx
    92ea:	89 fb                	mov    %edi,%ebx
    92ec:	48 83 ec 08          	sub    $0x8,%rsp
    92f0:	43 83 3c ae ff       	cmpl   $0xffffffff,(%r14,%r13,4)
    92f5:	0f 84 cd 00 00 00    	je     93c8 <cauchy_n_ones+0x108>
    92fb:	85 ed                	test   %ebp,%ebp
    92fd:	0f 8e 2c 01 00 00    	jle    942f <cauchy_n_ones+0x16f>
    9303:	31 c0                	xor    %eax,%eax
    9305:	31 c9                	xor    %ecx,%ecx
    9307:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    930e:	00 00 
    9310:	c4 e2 7a f7 d3       	sarx   %eax,%ebx,%edx
    9315:	83 e2 01             	and    $0x1,%edx
    9318:	83 fa 01             	cmp    $0x1,%edx
    931b:	83 d9 ff             	sbb    $0xffffffff,%ecx
    931e:	ff c0                	inc    %eax
    9320:	39 c5                	cmp    %eax,%ebp
    9322:	75 ec                	jne    9310 <cauchy_n_ones+0x50>
    9324:	83 fd 01             	cmp    $0x1,%ebp
    9327:	0f 8e 0e 01 00 00    	jle    943b <cauchy_n_ones+0x17b>
    932d:	4d 69 d5 84 00 00 00 	imul   $0x84,%r13,%r10
    9334:	4d 89 e9             	mov    %r13,%r9
    9337:	48 8d 05 22 84 00 00 	lea    0x8422(%rip),%rax        # 11760 <ONEs>
    933e:	49 c1 e1 05          	shl    $0x5,%r9
    9342:	49 01 c2             	add    %rax,%r10
    9345:	4d 01 e9             	add    %r13,%r9
    9348:	41 89 c8             	mov    %ecx,%r8d
    934b:	be 01 00 00 00       	mov    $0x1,%esi
    9350:	4c 8d 1d 29 95 00 00 	lea    0x9529(%rip),%r11        # 12880 <NOs>
    9357:	4c 8d 78 04          	lea    0x4(%rax),%r15
    935b:	eb 0e                	jmp    936b <cauchy_n_ones+0xab>
    935d:	0f 1f 00             	nopl   (%rax)
    9360:	01 db                	add    %ebx,%ebx
    9362:	ff c6                	inc    %esi
    9364:	41 01 c8             	add    %ecx,%r8d
    9367:	39 f5                	cmp    %esi,%ebp
    9369:	74 45                	je     93b0 <cauchy_n_ones+0xf0>
    936b:	41 85 dc             	test   %ebx,%r12d
    936e:	74 f0                	je     9360 <cauchy_n_ones+0xa0>
    9370:	44 31 e3             	xor    %r12d,%ebx
    9373:	43 8b 04 ab          	mov    (%r11,%r13,4),%eax
    9377:	01 db                	add    %ebx,%ebx
    9379:	43 33 1c ae          	xor    (%r14,%r13,4),%ebx
    937d:	ff c9                	dec    %ecx
    937f:	85 c0                	test   %eax,%eax
    9381:	7e df                	jle    9362 <cauchy_n_ones+0xa2>
    9383:	ff c8                	dec    %eax
    9385:	4c 01 c8             	add    %r9,%rax
    9388:	49 8d 3c 87          	lea    (%r15,%rax,4),%rdi
    938c:	4c 89 d0             	mov    %r10,%rax
    938f:	90                   	nop
    9390:	8b 10                	mov    (%rax),%edx
    9392:	21 da                	and    %ebx,%edx
    9394:	83 fa 01             	cmp    $0x1,%edx
    9397:	19 d2                	sbb    %edx,%edx
    9399:	83 ca 01             	or     $0x1,%edx
    939c:	48 83 c0 04          	add    $0x4,%rax
    93a0:	01 d1                	add    %edx,%ecx
    93a2:	48 39 f8             	cmp    %rdi,%rax
    93a5:	75 e9                	jne    9390 <cauchy_n_ones+0xd0>
    93a7:	ff c6                	inc    %esi
    93a9:	41 01 c8             	add    %ecx,%r8d
    93ac:	39 f5                	cmp    %esi,%ebp
    93ae:	75 bb                	jne    936b <cauchy_n_ones+0xab>
    93b0:	48 83 c4 08          	add    $0x8,%rsp
    93b4:	5b                   	pop    %rbx
    93b5:	5d                   	pop    %rbp
    93b6:	41 5c                	pop    %r12
    93b8:	41 5d                	pop    %r13
    93ba:	41 5e                	pop    %r14
    93bc:	44 89 c0             	mov    %r8d,%eax
    93bf:	41 5f                	pop    %r15
    93c1:	c3                   	retq   
    93c2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    93c8:	44 89 ea             	mov    %r13d,%edx
    93cb:	be 02 00 00 00       	mov    $0x2,%esi
    93d0:	44 89 e7             	mov    %r12d,%edi
    93d3:	e8 88 f2 ff ff       	callq  8660 <galois_single_multiply>
    93d8:	43 89 04 ae          	mov    %eax,(%r14,%r13,4)
    93dc:	45 85 ed             	test   %r13d,%r13d
    93df:	7e 56                	jle    9437 <cauchy_n_ones+0x177>
    93e1:	4d 89 e8             	mov    %r13,%r8
    93e4:	49 c1 e0 05          	shl    $0x5,%r8
    93e8:	31 d2                	xor    %edx,%edx
    93ea:	31 f6                	xor    %esi,%esi
    93ec:	4c 8d 15 6d 83 00 00 	lea    0x836d(%rip),%r10        # 11760 <ONEs>
    93f3:	4d 01 e8             	add    %r13,%r8
    93f6:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    93fc:	0f 1f 40 00          	nopl   0x0(%rax)
    9400:	0f a3 d0             	bt     %edx,%eax
    9403:	73 11                	jae    9416 <cauchy_n_ones+0x156>
    9405:	48 63 ce             	movslq %esi,%rcx
    9408:	4c 01 c1             	add    %r8,%rcx
    940b:	c4 c2 69 f7 f9       	shlx   %edx,%r9d,%edi
    9410:	41 89 3c 8a          	mov    %edi,(%r10,%rcx,4)
    9414:	ff c6                	inc    %esi
    9416:	ff c2                	inc    %edx
    9418:	39 d5                	cmp    %edx,%ebp
    941a:	75 e4                	jne    9400 <cauchy_n_ones+0x140>
    941c:	48 8d 05 5d 94 00 00 	lea    0x945d(%rip),%rax        # 12880 <NOs>
    9423:	42 89 34 a8          	mov    %esi,(%rax,%r13,4)
    9427:	85 ed                	test   %ebp,%ebp
    9429:	0f 8f d4 fe ff ff    	jg     9303 <cauchy_n_ones+0x43>
    942f:	45 31 c0             	xor    %r8d,%r8d
    9432:	e9 79 ff ff ff       	jmpq   93b0 <cauchy_n_ones+0xf0>
    9437:	31 f6                	xor    %esi,%esi
    9439:	eb e1                	jmp    941c <cauchy_n_ones+0x15c>
    943b:	41 89 c8             	mov    %ecx,%r8d
    943e:	e9 6d ff ff ff       	jmpq   93b0 <cauchy_n_ones+0xf0>
    9443:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    944a:	00 00 00 00 
    944e:	66 90                	xchg   %ax,%ax

0000000000009450 <cauchy_original_coding_matrix>:
    9450:	f3 0f 1e fa          	endbr64 
    9454:	41 57                	push   %r15
    9456:	41 89 ff             	mov    %edi,%r15d
    9459:	41 56                	push   %r14
    945b:	41 55                	push   %r13
    945d:	41 54                	push   %r12
    945f:	55                   	push   %rbp
    9460:	89 d5                	mov    %edx,%ebp
    9462:	53                   	push   %rbx
    9463:	48 83 ec 18          	sub    $0x18,%rsp
    9467:	89 34 24             	mov    %esi,(%rsp)
    946a:	83 fa 1e             	cmp    $0x1e,%edx
    946d:	7f 15                	jg     9484 <cauchy_original_coding_matrix+0x34>
    946f:	b8 01 00 00 00       	mov    $0x1,%eax
    9474:	8d 14 37             	lea    (%rdi,%rsi,1),%edx
    9477:	c4 e2 51 f7 c0       	shlx   %ebp,%eax,%eax
    947c:	39 c2                	cmp    %eax,%edx
    947e:	0f 8f 9d 00 00 00    	jg     9521 <cauchy_original_coding_matrix+0xd1>
    9484:	44 8b 34 24          	mov    (%rsp),%r14d
    9488:	44 89 f7             	mov    %r14d,%edi
    948b:	41 0f af ff          	imul   %r15d,%edi
    948f:	48 63 ff             	movslq %edi,%rdi
    9492:	48 c1 e7 02          	shl    $0x2,%rdi
    9496:	e8 65 7f ff ff       	callq  1400 <malloc@plt>
    949b:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    94a0:	48 85 c0             	test   %rax,%rax
    94a3:	74 7c                	je     9521 <cauchy_original_coding_matrix+0xd1>
    94a5:	c7 44 24 04 00 00 00 	movl   $0x0,0x4(%rsp)
    94ac:	00 
    94ad:	31 db                	xor    %ebx,%ebx
    94af:	47 8d 24 3e          	lea    (%r14,%r15,1),%r12d
    94b3:	45 85 f6             	test   %r14d,%r14d
    94b6:	7e 55                	jle    950d <cauchy_original_coding_matrix+0xbd>
    94b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    94bf:	00 
    94c0:	45 85 ff             	test   %r15d,%r15d
    94c3:	7e 41                	jle    9506 <cauchy_original_coding_matrix+0xb6>
    94c5:	48 63 44 24 04       	movslq 0x4(%rsp),%rax
    94ca:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    94cf:	44 8b 34 24          	mov    (%rsp),%r14d
    94d3:	4c 8d 2c 81          	lea    (%rcx,%rax,4),%r13
    94d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    94de:	00 00 
    94e0:	44 89 f6             	mov    %r14d,%esi
    94e3:	31 de                	xor    %ebx,%esi
    94e5:	89 ea                	mov    %ebp,%edx
    94e7:	bf 01 00 00 00       	mov    $0x1,%edi
    94ec:	e8 8f f1 ff ff       	callq  8680 <galois_single_divide>
    94f1:	41 ff c6             	inc    %r14d
    94f4:	41 89 45 00          	mov    %eax,0x0(%r13)
    94f8:	49 83 c5 04          	add    $0x4,%r13
    94fc:	45 39 e6             	cmp    %r12d,%r14d
    94ff:	75 df                	jne    94e0 <cauchy_original_coding_matrix+0x90>
    9501:	44 01 7c 24 04       	add    %r15d,0x4(%rsp)
    9506:	ff c3                	inc    %ebx
    9508:	39 1c 24             	cmp    %ebx,(%rsp)
    950b:	75 b3                	jne    94c0 <cauchy_original_coding_matrix+0x70>
    950d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    9512:	48 83 c4 18          	add    $0x18,%rsp
    9516:	5b                   	pop    %rbx
    9517:	5d                   	pop    %rbp
    9518:	41 5c                	pop    %r12
    951a:	41 5d                	pop    %r13
    951c:	41 5e                	pop    %r14
    951e:	41 5f                	pop    %r15
    9520:	c3                   	retq   
    9521:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    9528:	00 00 
    952a:	eb e1                	jmp    950d <cauchy_original_coding_matrix+0xbd>
    952c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000009530 <cauchy_xy_coding_matrix>:
    9530:	f3 0f 1e fa          	endbr64 
    9534:	41 57                	push   %r15
    9536:	49 89 cf             	mov    %rcx,%r15
    9539:	41 56                	push   %r14
    953b:	4d 89 c6             	mov    %r8,%r14
    953e:	41 55                	push   %r13
    9540:	41 89 fd             	mov    %edi,%r13d
    9543:	0f af fe             	imul   %esi,%edi
    9546:	41 54                	push   %r12
    9548:	41 89 f4             	mov    %esi,%r12d
    954b:	48 63 ff             	movslq %edi,%rdi
    954e:	55                   	push   %rbp
    954f:	48 c1 e7 02          	shl    $0x2,%rdi
    9553:	53                   	push   %rbx
    9554:	89 d3                	mov    %edx,%ebx
    9556:	48 83 ec 28          	sub    $0x28,%rsp
    955a:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
    955f:	e8 9c 7e ff ff       	callq  1400 <malloc@plt>
    9564:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    9569:	48 85 c0             	test   %rax,%rax
    956c:	74 75                	je     95e3 <cauchy_xy_coding_matrix+0xb3>
    956e:	45 85 e4             	test   %r12d,%r12d
    9571:	7e 70                	jle    95e3 <cauchy_xy_coding_matrix+0xb3>
    9573:	41 8d 44 24 ff       	lea    -0x1(%r12),%eax
    9578:	49 8d 44 87 04       	lea    0x4(%r15,%rax,4),%rax
    957d:	c7 44 24 04 00 00 00 	movl   $0x0,0x4(%rsp)
    9584:	00 
    9585:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    958a:	41 8d 45 ff          	lea    -0x1(%r13),%eax
    958e:	49 8d 6c 86 04       	lea    0x4(%r14,%rax,4),%rbp
    9593:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    9598:	45 85 ed             	test   %r13d,%r13d
    959b:	7e 3b                	jle    95d8 <cauchy_xy_coding_matrix+0xa8>
    959d:	48 63 44 24 04       	movslq 0x4(%rsp),%rax
    95a2:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    95a7:	4c 8b 74 24 18       	mov    0x18(%rsp),%r14
    95ac:	4c 8d 24 81          	lea    (%rcx,%rax,4),%r12
    95b0:	41 8b 37             	mov    (%r15),%esi
    95b3:	89 da                	mov    %ebx,%edx
    95b5:	41 33 36             	xor    (%r14),%esi
    95b8:	bf 01 00 00 00       	mov    $0x1,%edi
    95bd:	e8 be f0 ff ff       	callq  8680 <galois_single_divide>
    95c2:	49 83 c6 04          	add    $0x4,%r14
    95c6:	41 89 04 24          	mov    %eax,(%r12)
    95ca:	49 83 c4 04          	add    $0x4,%r12
    95ce:	49 39 ee             	cmp    %rbp,%r14
    95d1:	75 dd                	jne    95b0 <cauchy_xy_coding_matrix+0x80>
    95d3:	44 01 6c 24 04       	add    %r13d,0x4(%rsp)
    95d8:	49 83 c7 04          	add    $0x4,%r15
    95dc:	4c 3b 7c 24 10       	cmp    0x10(%rsp),%r15
    95e1:	75 b5                	jne    9598 <cauchy_xy_coding_matrix+0x68>
    95e3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    95e8:	48 83 c4 28          	add    $0x28,%rsp
    95ec:	5b                   	pop    %rbx
    95ed:	5d                   	pop    %rbp
    95ee:	41 5c                	pop    %r12
    95f0:	41 5d                	pop    %r13
    95f2:	41 5e                	pop    %r14
    95f4:	41 5f                	pop    %r15
    95f6:	c3                   	retq   
    95f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    95fe:	00 00 

0000000000009600 <cauchy_improve_coding_matrix>:
    9600:	f3 0f 1e fa          	endbr64 
    9604:	41 57                	push   %r15
    9606:	41 56                	push   %r14
    9608:	41 55                	push   %r13
    960a:	41 54                	push   %r12
    960c:	55                   	push   %rbp
    960d:	53                   	push   %rbx
    960e:	89 d3                	mov    %edx,%ebx
    9610:	48 83 ec 48          	sub    $0x48,%rsp
    9614:	89 7c 24 30          	mov    %edi,0x30(%rsp)
    9618:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
    961c:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
    9621:	85 ff                	test   %edi,%edi
    9623:	0f 8e 7e 00 00 00    	jle    96a7 <cauchy_improve_coding_matrix+0xa7>
    9629:	8d 47 ff             	lea    -0x1(%rdi),%eax
    962c:	4c 63 ef             	movslq %edi,%r13
    962f:	48 89 04 24          	mov    %rax,(%rsp)
    9633:	49 c1 e5 02          	shl    $0x2,%r13
    9637:	31 ed                	xor    %ebp,%ebp
    9639:	eb 12                	jmp    964d <cauchy_improve_coding_matrix+0x4d>
    963b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    9640:	48 8d 45 01          	lea    0x1(%rbp),%rax
    9644:	48 3b 2c 24          	cmp    (%rsp),%rbp
    9648:	74 5d                	je     96a7 <cauchy_improve_coding_matrix+0xa7>
    964a:	48 89 c5             	mov    %rax,%rbp
    964d:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9652:	8b 34 a8             	mov    (%rax,%rbp,4),%esi
    9655:	83 fe 01             	cmp    $0x1,%esi
    9658:	74 e6                	je     9640 <cauchy_improve_coding_matrix+0x40>
    965a:	89 da                	mov    %ebx,%edx
    965c:	bf 01 00 00 00       	mov    $0x1,%edi
    9661:	e8 1a f0 ff ff       	callq  8680 <galois_single_divide>
    9666:	8b 54 24 1c          	mov    0x1c(%rsp),%edx
    966a:	41 89 c4             	mov    %eax,%r12d
    966d:	85 d2                	test   %edx,%edx
    966f:	7e cf                	jle    9640 <cauchy_improve_coding_matrix+0x40>
    9671:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9676:	45 31 ff             	xor    %r15d,%r15d
    9679:	4c 8d 34 a8          	lea    (%rax,%rbp,4),%r14
    967d:	0f 1f 00             	nopl   (%rax)
    9680:	41 8b 3e             	mov    (%r14),%edi
    9683:	89 da                	mov    %ebx,%edx
    9685:	44 89 e6             	mov    %r12d,%esi
    9688:	e8 d3 ef ff ff       	callq  8660 <galois_single_multiply>
    968d:	41 ff c7             	inc    %r15d
    9690:	41 89 06             	mov    %eax,(%r14)
    9693:	4d 01 ee             	add    %r13,%r14
    9696:	44 39 7c 24 1c       	cmp    %r15d,0x1c(%rsp)
    969b:	75 e3                	jne    9680 <cauchy_improve_coding_matrix+0x80>
    969d:	48 8d 45 01          	lea    0x1(%rbp),%rax
    96a1:	48 3b 2c 24          	cmp    (%rsp),%rbp
    96a5:	75 a3                	jne    964a <cauchy_improve_coding_matrix+0x4a>
    96a7:	83 7c 24 1c 01       	cmpl   $0x1,0x1c(%rsp)
    96ac:	0f 8e 3b 01 00 00    	jle    97ed <cauchy_improve_coding_matrix+0x1ed>
    96b2:	48 63 44 24 30       	movslq 0x30(%rsp),%rax
    96b7:	c7 44 24 20 01 00 00 	movl   $0x1,0x20(%rsp)
    96be:	00 
    96bf:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    96c6:	00 
    96c7:	48 89 d7             	mov    %rdx,%rdi
    96ca:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
    96cf:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    96d4:	48 89 c1             	mov    %rax,%rcx
    96d7:	48 89 d6             	mov    %rdx,%rsi
    96da:	48 01 fe             	add    %rdi,%rsi
    96dd:	48 89 34 24          	mov    %rsi,(%rsp)
    96e1:	8d 70 ff             	lea    -0x1(%rax),%esi
    96e4:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
    96e9:	89 4c 24 34          	mov    %ecx,0x34(%rsp)
    96ed:	48 01 f0             	add    %rsi,%rax
    96f0:	4c 8d 7c 82 04       	lea    0x4(%rdx,%rax,4),%r15
    96f5:	0f 1f 00             	nopl   (%rax)
    96f8:	8b 44 24 30          	mov    0x30(%rsp),%eax
    96fc:	85 c0                	test   %eax,%eax
    96fe:	0f 8e c3 00 00 00    	jle    97c7 <cauchy_improve_coding_matrix+0x1c7>
    9704:	48 8b 2c 24          	mov    (%rsp),%rbp
    9708:	45 31 e4             	xor    %r12d,%r12d
    970b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    9710:	8b 7d 00             	mov    0x0(%rbp),%edi
    9713:	89 de                	mov    %ebx,%esi
    9715:	e8 a6 fb ff ff       	callq  92c0 <cauchy_n_ones>
    971a:	48 83 c5 04          	add    $0x4,%rbp
    971e:	41 01 c4             	add    %eax,%r12d
    9721:	4c 39 fd             	cmp    %r15,%rbp
    9724:	75 ea                	jne    9710 <cauchy_improve_coding_matrix+0x110>
    9726:	44 89 64 24 18       	mov    %r12d,0x18(%rsp)
    972b:	c7 44 24 24 ff ff ff 	movl   $0xffffffff,0x24(%rsp)
    9732:	ff 
    9733:	45 31 f6             	xor    %r14d,%r14d
    9736:	eb 16                	jmp    974e <cauchy_improve_coding_matrix+0x14e>
    9738:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    973f:	00 
    9740:	49 8d 46 01          	lea    0x1(%r14),%rax
    9744:	4c 3b 74 24 10       	cmp    0x10(%rsp),%r14
    9749:	74 75                	je     97c0 <cauchy_improve_coding_matrix+0x1c0>
    974b:	49 89 c6             	mov    %rax,%r14
    974e:	48 8b 04 24          	mov    (%rsp),%rax
    9752:	44 89 74 24 0c       	mov    %r14d,0xc(%rsp)
    9757:	42 8b 34 b0          	mov    (%rax,%r14,4),%esi
    975b:	83 fe 01             	cmp    $0x1,%esi
    975e:	74 e0                	je     9740 <cauchy_improve_coding_matrix+0x140>
    9760:	89 da                	mov    %ebx,%edx
    9762:	bf 01 00 00 00       	mov    $0x1,%edi
    9767:	e8 14 ef ff ff       	callq  8680 <galois_single_divide>
    976c:	4c 8b 2c 24          	mov    (%rsp),%r13
    9770:	41 89 c4             	mov    %eax,%r12d
    9773:	31 ed                	xor    %ebp,%ebp
    9775:	0f 1f 00             	nopl   (%rax)
    9778:	41 8b 7d 00          	mov    0x0(%r13),%edi
    977c:	89 da                	mov    %ebx,%edx
    977e:	44 89 e6             	mov    %r12d,%esi
    9781:	e8 da ee ff ff       	callq  8660 <galois_single_multiply>
    9786:	89 c7                	mov    %eax,%edi
    9788:	89 de                	mov    %ebx,%esi
    978a:	e8 31 fb ff ff       	callq  92c0 <cauchy_n_ones>
    978f:	49 83 c5 04          	add    $0x4,%r13
    9793:	01 c5                	add    %eax,%ebp
    9795:	4d 39 fd             	cmp    %r15,%r13
    9798:	75 de                	jne    9778 <cauchy_improve_coding_matrix+0x178>
    979a:	3b 6c 24 18          	cmp    0x18(%rsp),%ebp
    979e:	7d a0                	jge    9740 <cauchy_improve_coding_matrix+0x140>
    97a0:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    97a4:	89 6c 24 18          	mov    %ebp,0x18(%rsp)
    97a8:	89 44 24 24          	mov    %eax,0x24(%rsp)
    97ac:	49 8d 46 01          	lea    0x1(%r14),%rax
    97b0:	4c 3b 74 24 10       	cmp    0x10(%rsp),%r14
    97b5:	75 94                	jne    974b <cauchy_improve_coding_matrix+0x14b>
    97b7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    97be:	00 00 
    97c0:	83 7c 24 24 ff       	cmpl   $0xffffffff,0x24(%rsp)
    97c5:	75 39                	jne    9800 <cauchy_improve_coding_matrix+0x200>
    97c7:	ff 44 24 20          	incl   0x20(%rsp)
    97cb:	48 8b 54 24 38       	mov    0x38(%rsp),%rdx
    97d0:	8b 74 24 30          	mov    0x30(%rsp),%esi
    97d4:	48 01 14 24          	add    %rdx,(%rsp)
    97d8:	8b 44 24 20          	mov    0x20(%rsp),%eax
    97dc:	01 74 24 34          	add    %esi,0x34(%rsp)
    97e0:	49 01 d7             	add    %rdx,%r15
    97e3:	39 44 24 1c          	cmp    %eax,0x1c(%rsp)
    97e7:	0f 85 0b ff ff ff    	jne    96f8 <cauchy_improve_coding_matrix+0xf8>
    97ed:	48 83 c4 48          	add    $0x48,%rsp
    97f1:	5b                   	pop    %rbx
    97f2:	5d                   	pop    %rbp
    97f3:	41 5c                	pop    %r12
    97f5:	41 5d                	pop    %r13
    97f7:	41 5e                	pop    %r14
    97f9:	41 5f                	pop    %r15
    97fb:	c3                   	retq   
    97fc:	0f 1f 40 00          	nopl   0x0(%rax)
    9800:	8b 44 24 24          	mov    0x24(%rsp),%eax
    9804:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
    9809:	03 44 24 34          	add    0x34(%rsp),%eax
    980d:	48 98                	cltq   
    980f:	8b 34 81             	mov    (%rcx,%rax,4),%esi
    9812:	89 da                	mov    %ebx,%edx
    9814:	bf 01 00 00 00       	mov    $0x1,%edi
    9819:	e8 62 ee ff ff       	callq  8680 <galois_single_divide>
    981e:	4c 8b 24 24          	mov    (%rsp),%r12
    9822:	89 c5                	mov    %eax,%ebp
    9824:	0f 1f 40 00          	nopl   0x0(%rax)
    9828:	41 8b 3c 24          	mov    (%r12),%edi
    982c:	89 da                	mov    %ebx,%edx
    982e:	89 ee                	mov    %ebp,%esi
    9830:	e8 2b ee ff ff       	callq  8660 <galois_single_multiply>
    9835:	41 89 04 24          	mov    %eax,(%r12)
    9839:	49 83 c4 04          	add    $0x4,%r12
    983d:	4d 39 fc             	cmp    %r15,%r12
    9840:	75 e6                	jne    9828 <cauchy_improve_coding_matrix+0x228>
    9842:	eb 83                	jmp    97c7 <cauchy_improve_coding_matrix+0x1c7>
    9844:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    984b:	00 00 00 00 
    984f:	90                   	nop

0000000000009850 <cauchy_good_general_coding_matrix>:
    9850:	f3 0f 1e fa          	endbr64 
    9854:	41 56                	push   %r14
    9856:	41 89 fe             	mov    %edi,%r14d
    9859:	41 55                	push   %r13
    985b:	41 89 d5             	mov    %edx,%r13d
    985e:	41 54                	push   %r12
    9860:	55                   	push   %rbp
    9861:	89 f5                	mov    %esi,%ebp
    9863:	53                   	push   %rbx
    9864:	83 fe 02             	cmp    $0x2,%esi
    9867:	75 0f                	jne    9878 <cauchy_good_general_coding_matrix+0x28>
    9869:	48 63 da             	movslq %edx,%rbx
    986c:	48 8d 05 6d 15 00 00 	lea    0x156d(%rip),%rax        # ade0 <cbest_max_k>
    9873:	39 3c 98             	cmp    %edi,(%rax,%rbx,4)
    9876:	7d 38                	jge    98b0 <cauchy_good_general_coding_matrix+0x60>
    9878:	44 89 ea             	mov    %r13d,%edx
    987b:	89 ee                	mov    %ebp,%esi
    987d:	44 89 f7             	mov    %r14d,%edi
    9880:	e8 cb fb ff ff       	callq  9450 <cauchy_original_coding_matrix>
    9885:	49 89 c4             	mov    %rax,%r12
    9888:	48 85 c0             	test   %rax,%rax
    988b:	74 10                	je     989d <cauchy_good_general_coding_matrix+0x4d>
    988d:	48 89 c1             	mov    %rax,%rcx
    9890:	44 89 ea             	mov    %r13d,%edx
    9893:	89 ee                	mov    %ebp,%esi
    9895:	44 89 f7             	mov    %r14d,%edi
    9898:	e8 63 fd ff ff       	callq  9600 <cauchy_improve_coding_matrix>
    989d:	5b                   	pop    %rbx
    989e:	5d                   	pop    %rbp
    989f:	4c 89 e0             	mov    %r12,%rax
    98a2:	41 5c                	pop    %r12
    98a4:	41 5d                	pop    %r13
    98a6:	41 5e                	pop    %r14
    98a8:	c3                   	retq   
    98a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    98b0:	8d 3c 3f             	lea    (%rdi,%rdi,1),%edi
    98b3:	48 63 ff             	movslq %edi,%rdi
    98b6:	48 c1 e7 02          	shl    $0x2,%rdi
    98ba:	e8 41 7b ff ff       	callq  1400 <malloc@plt>
    98bf:	49 89 c4             	mov    %rax,%r12
    98c2:	48 85 c0             	test   %rax,%rax
    98c5:	74 d6                	je     989d <cauchy_good_general_coding_matrix+0x4d>
    98c7:	8b 05 7b 7e 00 00    	mov    0x7e7b(%rip),%eax        # 11748 <cbest_init>
    98cd:	85 c0                	test   %eax,%eax
    98cf:	0f 85 93 01 00 00    	jne    9a68 <cauchy_good_general_coding_matrix+0x218>
    98d5:	48 8d 05 a4 77 00 00 	lea    0x77a4(%rip),%rax        # 11080 <cbest_2>
    98dc:	48 89 05 6d 7d 00 00 	mov    %rax,0x7d6d(%rip)        # 11650 <cbest_all+0x10>
    98e3:	48 8d 05 76 77 00 00 	lea    0x7776(%rip),%rax        # 11060 <cbest_3>
    98ea:	48 89 05 67 7d 00 00 	mov    %rax,0x7d67(%rip)        # 11658 <cbest_all+0x18>
    98f1:	48 8d 05 28 77 00 00 	lea    0x7728(%rip),%rax        # 11020 <cbest_4>
    98f8:	48 89 05 61 7d 00 00 	mov    %rax,0x7d61(%rip)        # 11660 <cbest_all+0x20>
    98ff:	48 8d 05 9a 76 00 00 	lea    0x769a(%rip),%rax        # 10fa0 <cbest_5>
    9906:	48 89 05 5b 7d 00 00 	mov    %rax,0x7d5b(%rip)        # 11668 <cbest_all+0x28>
    990d:	48 8d 05 8c 75 00 00 	lea    0x758c(%rip),%rax        # 10ea0 <cbest_6>
    9914:	48 89 05 55 7d 00 00 	mov    %rax,0x7d55(%rip)        # 11670 <cbest_all+0x30>
    991b:	48 8d 05 7e 73 00 00 	lea    0x737e(%rip),%rax        # 10ca0 <cbest_7>
    9922:	48 89 05 4f 7d 00 00 	mov    %rax,0x7d4f(%rip)        # 11678 <cbest_all+0x38>
    9929:	48 8d 05 70 6f 00 00 	lea    0x6f70(%rip),%rax        # 108a0 <cbest_8>
    9930:	48 89 05 49 7d 00 00 	mov    %rax,0x7d49(%rip)        # 11680 <cbest_all+0x40>
    9937:	48 8d 05 62 67 00 00 	lea    0x6762(%rip),%rax        # 100a0 <cbest_9>
    993e:	48 89 05 43 7d 00 00 	mov    %rax,0x7d43(%rip)        # 11688 <cbest_all+0x48>
    9945:	48 8d 05 54 57 00 00 	lea    0x5754(%rip),%rax        # f0a0 <cbest_10>
    994c:	48 89 05 3d 7d 00 00 	mov    %rax,0x7d3d(%rip)        # 11690 <cbest_all+0x50>
    9953:	48 8d 05 46 47 00 00 	lea    0x4746(%rip),%rax        # e0a0 <cbest_11>
    995a:	c7 05 e4 7d 00 00 01 	movl   $0x1,0x7de4(%rip)        # 11748 <cbest_init>
    9961:	00 00 00 
    9964:	48 c7 05 d1 7c 00 00 	movq   $0x0,0x7cd1(%rip)        # 11640 <cbest_all>
    996b:	00 00 00 00 
    996f:	48 c7 05 ce 7c 00 00 	movq   $0x0,0x7cce(%rip)        # 11648 <cbest_all+0x8>
    9976:	00 00 00 00 
    997a:	48 89 05 17 7d 00 00 	mov    %rax,0x7d17(%rip)        # 11698 <cbest_all+0x58>
    9981:	48 c7 05 14 7d 00 00 	movq   $0x0,0x7d14(%rip)        # 116a0 <cbest_all+0x60>
    9988:	00 00 00 00 
    998c:	48 c7 05 11 7d 00 00 	movq   $0x0,0x7d11(%rip)        # 116a8 <cbest_all+0x68>
    9993:	00 00 00 00 
    9997:	48 c7 05 0e 7d 00 00 	movq   $0x0,0x7d0e(%rip)        # 116b0 <cbest_all+0x70>
    999e:	00 00 00 00 
    99a2:	48 c7 05 0b 7d 00 00 	movq   $0x0,0x7d0b(%rip)        # 116b8 <cbest_all+0x78>
    99a9:	00 00 00 00 
    99ad:	48 c7 05 08 7d 00 00 	movq   $0x0,0x7d08(%rip)        # 116c0 <cbest_all+0x80>
    99b4:	00 00 00 00 
    99b8:	48 c7 05 05 7d 00 00 	movq   $0x0,0x7d05(%rip)        # 116c8 <cbest_all+0x88>
    99bf:	00 00 00 00 
    99c3:	48 c7 05 02 7d 00 00 	movq   $0x0,0x7d02(%rip)        # 116d0 <cbest_all+0x90>
    99ca:	00 00 00 00 
    99ce:	48 c7 05 ff 7c 00 00 	movq   $0x0,0x7cff(%rip)        # 116d8 <cbest_all+0x98>
    99d5:	00 00 00 00 
    99d9:	48 c7 05 fc 7c 00 00 	movq   $0x0,0x7cfc(%rip)        # 116e0 <cbest_all+0xa0>
    99e0:	00 00 00 00 
    99e4:	48 c7 05 f9 7c 00 00 	movq   $0x0,0x7cf9(%rip)        # 116e8 <cbest_all+0xa8>
    99eb:	00 00 00 00 
    99ef:	48 c7 05 f6 7c 00 00 	movq   $0x0,0x7cf6(%rip)        # 116f0 <cbest_all+0xb0>
    99f6:	00 00 00 00 
    99fa:	48 c7 05 f3 7c 00 00 	movq   $0x0,0x7cf3(%rip)        # 116f8 <cbest_all+0xb8>
    9a01:	00 00 00 00 
    9a05:	48 c7 05 f0 7c 00 00 	movq   $0x0,0x7cf0(%rip)        # 11700 <cbest_all+0xc0>
    9a0c:	00 00 00 00 
    9a10:	48 c7 05 ed 7c 00 00 	movq   $0x0,0x7ced(%rip)        # 11708 <cbest_all+0xc8>
    9a17:	00 00 00 00 
    9a1b:	48 c7 05 ea 7c 00 00 	movq   $0x0,0x7cea(%rip)        # 11710 <cbest_all+0xd0>
    9a22:	00 00 00 00 
    9a26:	48 c7 05 e7 7c 00 00 	movq   $0x0,0x7ce7(%rip)        # 11718 <cbest_all+0xd8>
    9a2d:	00 00 00 00 
    9a31:	48 c7 05 e4 7c 00 00 	movq   $0x0,0x7ce4(%rip)        # 11720 <cbest_all+0xe0>
    9a38:	00 00 00 00 
    9a3c:	48 c7 05 e1 7c 00 00 	movq   $0x0,0x7ce1(%rip)        # 11728 <cbest_all+0xe8>
    9a43:	00 00 00 00 
    9a47:	48 c7 05 de 7c 00 00 	movq   $0x0,0x7cde(%rip)        # 11730 <cbest_all+0xf0>
    9a4e:	00 00 00 00 
    9a52:	48 c7 05 db 7c 00 00 	movq   $0x0,0x7cdb(%rip)        # 11738 <cbest_all+0xf8>
    9a59:	00 00 00 00 
    9a5d:	48 c7 05 d8 7c 00 00 	movq   $0x0,0x7cd8(%rip)        # 11740 <cbest_all+0x100>
    9a64:	00 00 00 00 
    9a68:	45 85 f6             	test   %r14d,%r14d
    9a6b:	0f 8e 2c fe ff ff    	jle    989d <cauchy_good_general_coding_matrix+0x4d>
    9a71:	48 8d 05 c8 7b 00 00 	lea    0x7bc8(%rip),%rax        # 11640 <cbest_all>
    9a78:	49 63 fe             	movslq %r14d,%rdi
    9a7b:	4c 8b 04 d8          	mov    (%rax,%rbx,8),%r8
    9a7f:	41 8d 76 ff          	lea    -0x1(%r14),%esi
    9a83:	49 8d 0c bc          	lea    (%r12,%rdi,4),%rcx
    9a87:	31 d2                	xor    %edx,%edx
    9a89:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    9a90:	41 c7 04 94 01 00 00 	movl   $0x1,(%r12,%rdx,4)
    9a97:	00 
    9a98:	41 8b 04 90          	mov    (%r8,%rdx,4),%eax
    9a9c:	89 04 91             	mov    %eax,(%rcx,%rdx,4)
    9a9f:	48 89 d0             	mov    %rdx,%rax
    9aa2:	48 ff c2             	inc    %rdx
    9aa5:	48 39 c6             	cmp    %rax,%rsi
    9aa8:	75 e6                	jne    9a90 <cauchy_good_general_coding_matrix+0x240>
    9aaa:	5b                   	pop    %rbx
    9aab:	5d                   	pop    %rbp
    9aac:	4c 89 e0             	mov    %r12,%rax
    9aaf:	41 5c                	pop    %r12
    9ab1:	41 5d                	pop    %r13
    9ab3:	41 5e                	pop    %r14
    9ab5:	c3                   	retq   
    9ab6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    9abd:	00 00 00 

0000000000009ac0 <timing_delta>:
    9ac0:	f3 0f 1e fa          	endbr64 
    9ac4:	41 57                	push   %r15
    9ac6:	49 89 f7             	mov    %rsi,%r15
    9ac9:	41 56                	push   %r14
    9acb:	41 55                	push   %r13
    9acd:	41 54                	push   %r12
    9acf:	55                   	push   %rbp
    9ad0:	53                   	push   %rbx
    9ad1:	48 83 ec 68          	sub    $0x68,%rsp
    9ad5:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
    9ada:	c5 fb 10 05 26 8e 00 	vmovsd 0x8e26(%rip),%xmm0        # 12908 <timing_time.2619>
    9ae1:	00 
    9ae2:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    9ae9:	00 00 
    9aeb:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    9af0:	31 c0                	xor    %eax,%eax
    9af2:	c5 f9 2e 05 e6 0b 00 	vucomisd 0xbe6(%rip),%xmm0        # a6e0 <__PRETTY_FUNCTION__.5741+0x7>
    9af9:	00 
    9afa:	7a 73                	jp     9b6f <timing_delta+0xaf>
    9afc:	75 71                	jne    9b6f <timing_delta+0xaf>
    9afe:	bb 10 27 00 00       	mov    $0x2710,%ebx
    9b03:	4c 8d 74 24 10       	lea    0x10(%rsp),%r14
    9b08:	4c 8d 6c 24 30       	lea    0x30(%rsp),%r13
    9b0d:	4c 8d 64 24 40       	lea    0x40(%rsp),%r12
    9b12:	48 8d 6c 24 20       	lea    0x20(%rsp),%rbp
    9b17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    9b1e:	00 00 
    9b20:	4c 89 f6             	mov    %r14,%rsi
    9b23:	31 ff                	xor    %edi,%edi
    9b25:	e8 b6 77 ff ff       	callq  12e0 <clock_gettime@plt>
    9b2a:	4c 89 ee             	mov    %r13,%rsi
    9b2d:	31 ff                	xor    %edi,%edi
    9b2f:	e8 ac 77 ff ff       	callq  12e0 <clock_gettime@plt>
    9b34:	4c 89 e6             	mov    %r12,%rsi
    9b37:	31 ff                	xor    %edi,%edi
    9b39:	e8 a2 77 ff ff       	callq  12e0 <clock_gettime@plt>
    9b3e:	48 89 ee             	mov    %rbp,%rsi
    9b41:	31 ff                	xor    %edi,%edi
    9b43:	e8 98 77 ff ff       	callq  12e0 <clock_gettime@plt>
    9b48:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9b4d:	c5 e1 57 db          	vxorpd %xmm3,%xmm3,%xmm3
    9b51:	48 2b 44 24 18       	sub    0x18(%rsp),%rax
    9b56:	c4 e1 e3 2a c0       	vcvtsi2sd %rax,%xmm3,%xmm0
    9b5b:	c5 fb 5e 05 05 13 00 	vdivsd 0x1305(%rip),%xmm0,%xmm0        # ae68 <cbest_max_k+0x88>
    9b62:	00 
    9b63:	c5 fb 11 05 9d 8d 00 	vmovsd %xmm0,0x8d9d(%rip)        # 12908 <timing_time.2619>
    9b6a:	00 
    9b6b:	ff cb                	dec    %ebx
    9b6d:	75 b1                	jne    9b20 <timing_delta+0x60>
    9b6f:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
    9b74:	49 8b 47 08          	mov    0x8(%r15),%rax
    9b78:	c5 d9 57 e4          	vxorpd %xmm4,%xmm4,%xmm4
    9b7c:	48 2b 42 08          	sub    0x8(%rdx),%rax
    9b80:	c4 e1 db 2a c8       	vcvtsi2sd %rax,%xmm4,%xmm1
    9b85:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
    9b8a:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    9b91:	00 00 
    9b93:	c5 f3 5c c0          	vsubsd %xmm0,%xmm1,%xmm0
    9b97:	c4 c1 db 2a 0f       	vcvtsi2sdq (%r15),%xmm4,%xmm1
    9b9c:	c5 fb 5e 05 cc 12 00 	vdivsd 0x12cc(%rip),%xmm0,%xmm0        # ae70 <cbest_max_k+0x90>
    9ba3:	00 
    9ba4:	c5 f9 28 d1          	vmovapd %xmm1,%xmm2
    9ba8:	c4 e1 db 2a 0a       	vcvtsi2sdq (%rdx),%xmm4,%xmm1
    9bad:	c5 eb 5c c9          	vsubsd %xmm1,%xmm2,%xmm1
    9bb1:	c5 fb 58 c1          	vaddsd %xmm1,%xmm0,%xmm0
    9bb5:	75 0f                	jne    9bc6 <timing_delta+0x106>
    9bb7:	48 83 c4 68          	add    $0x68,%rsp
    9bbb:	5b                   	pop    %rbx
    9bbc:	5d                   	pop    %rbp
    9bbd:	41 5c                	pop    %r12
    9bbf:	41 5d                	pop    %r13
    9bc1:	41 5e                	pop    %r14
    9bc3:	41 5f                	pop    %r15
    9bc5:	c3                   	retq   
    9bc6:	e8 45 77 ff ff       	callq  1310 <__stack_chk_fail@plt>
    9bcb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000009bd0 <__libc_csu_init>:
    9bd0:	f3 0f 1e fa          	endbr64 
    9bd4:	41 57                	push   %r15
    9bd6:	4c 8d 3d c3 40 00 00 	lea    0x40c3(%rip),%r15        # dca0 <__frame_dummy_init_array_entry>
    9bdd:	41 56                	push   %r14
    9bdf:	49 89 d6             	mov    %rdx,%r14
    9be2:	41 55                	push   %r13
    9be4:	49 89 f5             	mov    %rsi,%r13
    9be7:	41 54                	push   %r12
    9be9:	41 89 fc             	mov    %edi,%r12d
    9bec:	55                   	push   %rbp
    9bed:	48 8d 2d b4 40 00 00 	lea    0x40b4(%rip),%rbp        # dca8 <__do_global_dtors_aux_fini_array_entry>
    9bf4:	53                   	push   %rbx
    9bf5:	4c 29 fd             	sub    %r15,%rbp
    9bf8:	48 83 ec 08          	sub    $0x8,%rsp
    9bfc:	e8 ff 73 ff ff       	callq  1000 <_init>
    9c01:	48 c1 fd 03          	sar    $0x3,%rbp
    9c05:	74 1f                	je     9c26 <__libc_csu_init+0x56>
    9c07:	31 db                	xor    %ebx,%ebx
    9c09:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    9c10:	4c 89 f2             	mov    %r14,%rdx
    9c13:	4c 89 ee             	mov    %r13,%rsi
    9c16:	44 89 e7             	mov    %r12d,%edi
    9c19:	41 ff 14 df          	callq  *(%r15,%rbx,8)
    9c1d:	48 83 c3 01          	add    $0x1,%rbx
    9c21:	48 39 dd             	cmp    %rbx,%rbp
    9c24:	75 ea                	jne    9c10 <__libc_csu_init+0x40>
    9c26:	48 83 c4 08          	add    $0x8,%rsp
    9c2a:	5b                   	pop    %rbx
    9c2b:	5d                   	pop    %rbp
    9c2c:	41 5c                	pop    %r12
    9c2e:	41 5d                	pop    %r13
    9c30:	41 5e                	pop    %r14
    9c32:	41 5f                	pop    %r15
    9c34:	c3                   	retq   
    9c35:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    9c3c:	00 00 00 00 

0000000000009c40 <__libc_csu_fini>:
    9c40:	f3 0f 1e fa          	endbr64 
    9c44:	c3                   	retq   

Disassembly of section .fini:

0000000000009c48 <_fini>:
    9c48:	f3 0f 1e fa          	endbr64 
    9c4c:	48 83 ec 08          	sub    $0x8,%rsp
    9c50:	48 83 c4 08          	add    $0x8,%rsp
    9c54:	c3                   	retq   

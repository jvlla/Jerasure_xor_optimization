
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
    1d3d:	e8 be 45 00 00       	callq  6300 <jerasure_smart_bitmatrix_to_schedule>
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
    1d72:	e8 f9 7c 00 00       	callq  9a70 <timing_delta>
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
    1e57:	e8 04 42 00 00       	callq  6060 <jerasure_schedule_encode>
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
    1f4d:	e8 8e 6b 00 00       	callq  8ae0 <reed_sol_r6_encode>
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
    2082:	e8 e9 79 00 00       	callq  9a70 <timing_delta>
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
    2273:	e8 f8 77 00 00       	callq  9a70 <timing_delta>
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
    2364:	e8 97 74 00 00       	callq  9800 <cauchy_good_general_coding_matrix>
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
    23a3:	e8 58 3f 00 00       	callq  6300 <jerasure_smart_bitmatrix_to_schedule>
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
    2452:	e8 a9 6f 00 00       	callq  9400 <cauchy_original_coding_matrix>
    2457:	e9 0d ff ff ff       	jmpq   2369 <main+0xea9>
    245c:	48 8d 3d c7 7b 00 00 	lea    0x7bc7(%rip),%rdi        # a02a <_IO_stdin_used+0x2a>
    2463:	e8 38 f0 ff ff       	callq  14a0 <strdup@plt>
    2468:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    246d:	e9 4e f6 ff ff       	jmpq   1ac0 <main+0x600>
    2472:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    2479:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    2480:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    2487:	e8 44 6d 00 00       	callq  91d0 <reed_sol_vandermonde_coding_matrix>
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
    2793:	4c 8d 05 56 74 00 00 	lea    0x7456(%rip),%r8        # 9bf0 <__libc_csu_fini>
    279a:	48 8d 0d df 73 00 00 	lea    0x73df(%rip),%rcx        # 9b80 <__libc_csu_init>
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
    3744:	e8 c7 4e 00 00       	callq  8610 <galois_single_multiply>
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
    39c9:	e8 02 47 00 00       	callq  80d0 <galois_region_xor>
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
    3b7c:	e8 4f 45 00 00       	callq  80d0 <galois_region_xor>
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
    3d50:	e8 bb 48 00 00       	callq  8610 <galois_single_multiply>
    3d55:	43 31 04 2f          	xor    %eax,(%r15,%r13,1)
            inv[rs2+x] ^= galois_single_multiply(tmp, inv[row_start+x], w);
    3d59:	89 da                	mov    %ebx,%edx
    3d5b:	89 ef                	mov    %ebp,%edi
    3d5d:	43 8b 34 26          	mov    (%r14,%r12,1),%esi
    3d61:	49 83 c4 04          	add    $0x4,%r12
    3d65:	e8 a6 48 00 00       	callq  8610 <galois_single_multiply>
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
    3f00:	e8 2b 47 00 00       	callq  8630 <galois_single_divide>
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
    3f30:	e8 db 46 00 00       	callq  8610 <galois_single_multiply>
    3f35:	41 89 06             	mov    %eax,(%r14)
        inv[row_start+j] = galois_single_multiply(inv[row_start+j], inverse, w);
    3f38:	89 da                	mov    %ebx,%edx
    3f3a:	44 89 ee             	mov    %r13d,%esi
    3f3d:	41 8b 3c 24          	mov    (%r12),%edi
    3f41:	49 83 c6 04          	add    $0x4,%r14
    3f45:	e8 c6 46 00 00       	callq  8610 <galois_single_multiply>
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
    4039:	e8 d2 45 00 00       	callq  8610 <galois_single_multiply>
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
    431a:	e8 f1 42 00 00       	callq  8610 <galois_single_multiply>
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
    445b:	e8 d0 41 00 00       	callq  8630 <galois_single_divide>
    4460:	4c 8b 64 24 50       	mov    0x50(%rsp),%r12
    4465:	48 8b 5c 24 58       	mov    0x58(%rsp),%rbx
    446a:	89 c5                	mov    %eax,%ebp
      for (j = 0; j < cols; j++) { 
    446c:	0f 1f 40 00          	nopl   0x0(%rax)
        mat[row_start+j] = galois_single_multiply(mat[row_start+j], inverse, w);
    4470:	41 8b 3c 24          	mov    (%r12),%edi
    4474:	44 89 ea             	mov    %r13d,%edx
    4477:	89 ee                	mov    %ebp,%esi
    4479:	e8 92 41 00 00       	callq  8610 <galois_single_multiply>
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
    497d:	e8 9e 31 00 00       	callq  7b20 <galois_w16_region_multiply>
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
    49ee:	e8 dd 36 00 00       	callq  80d0 <galois_region_xor>
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
    4a58:	e8 43 2f 00 00       	callq  79a0 <galois_w08_region_multiply>
    4a5d:	e9 b1 fe ff ff       	jmpq   4913 <jerasure_matrix_dotprod+0x163>
    4a62:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
        case 32: galois_w32_region_multiply(sptr, matrix_row[i], size, dptr, init); break;
    4a68:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    4a6d:	44 89 f2             	mov    %r14d,%edx
    4a70:	e8 db 37 00 00       	callq  8250 <galois_w32_region_multiply>
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
    5c09:	e8 02 2a 00 00       	callq  8610 <galois_single_multiply>
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
    5cc6:	0f 8e 9a 02 00 00    	jle    5f66 <jerasure_do_scheduled_operations+0x2a6>
{
    5ccc:	41 54                	push   %r12
  {
    prev_dptr = NULL;
    for (op = 0; operations[op][0] >= 0; op++) {
    5cce:	45 31 d2             	xor    %r10d,%r10d
{
    5cd1:	55                   	push   %rbp
    5cd2:	53                   	push   %rbx
    for (op = 0; operations[op][0] >= 0; op++) {
    5cd3:	48 8b 1e             	mov    (%rsi),%rbx
    5cd6:	44 8b 1b             	mov    (%rbx),%r11d
    5cd9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    prev_dptr = NULL;
    5ce0:	31 c0                	xor    %eax,%eax
    for (op = 0; operations[op][0] >= 0; op++) {
    5ce2:	45 85 db             	test   %r11d,%r11d
    5ce5:	0f 88 f0 01 00 00    	js     5edb <jerasure_do_scheduled_operations+0x21b>
    5ceb:	4c 8d 4e 08          	lea    0x8(%rsi),%r9
    5cef:	49 63 eb             	movslq %r11d,%rbp
    5cf2:	49 89 d8             	mov    %rbx,%r8
    prev_dptr = NULL;
    5cf5:	31 c0                	xor    %eax,%eax
    5cf7:	e9 89 00 00 00       	jmpq   5d85 <jerasure_do_scheduled_operations+0xc5>
    5cfc:	0f 1f 40 00          	nopl   0x0(%rax)
      sptr = ptrs[operations[op][0]] + operations[op][1]*packetsize + i;
      dptr = ptrs[operations[op][2]] + operations[op][3]*packetsize + i;
      if (operations[op][4]) {
        // do xor operation here
        __asm__ __volatile__ (
    5d00:	c5 fd ef 01          	vpxor  (%rcx),%ymm0,%ymm0
    5d04:	c5 f5 ef 49 20       	vpxor  0x20(%rcx),%ymm1,%ymm1
    5d09:	c5 ed ef 51 40       	vpxor  0x40(%rcx),%ymm2,%ymm2
    5d0e:	c5 e5 ef 59 60       	vpxor  0x60(%rcx),%ymm3,%ymm3
    5d13:	c5 dd ef a1 80 00 00 	vpxor  0x80(%rcx),%ymm4,%ymm4
    5d1a:	00 
    5d1b:	c5 d5 ef a9 a0 00 00 	vpxor  0xa0(%rcx),%ymm5,%ymm5
    5d22:	00 
    5d23:	c5 cd ef b1 c0 00 00 	vpxor  0xc0(%rcx),%ymm6,%ymm6
    5d2a:	00 
    5d2b:	c5 c5 ef b9 e0 00 00 	vpxor  0xe0(%rcx),%ymm7,%ymm7
    5d32:	00 
    5d33:	c5 3d ef 81 00 01 00 	vpxor  0x100(%rcx),%ymm8,%ymm8
    5d3a:	00 
    5d3b:	c5 35 ef 89 20 01 00 	vpxor  0x120(%rcx),%ymm9,%ymm9
    5d42:	00 
    5d43:	c5 2d ef 91 40 01 00 	vpxor  0x140(%rcx),%ymm10,%ymm10
    5d4a:	00 
    5d4b:	c5 25 ef 99 60 01 00 	vpxor  0x160(%rcx),%ymm11,%ymm11
    5d52:	00 
    5d53:	c5 1d ef a1 80 01 00 	vpxor  0x180(%rcx),%ymm12,%ymm12
    5d5a:	00 
    5d5b:	c5 15 ef a9 a0 01 00 	vpxor  0x1a0(%rcx),%ymm13,%ymm13
    5d62:	00 
    5d63:	c5 0d ef b1 c0 01 00 	vpxor  0x1c0(%rcx),%ymm14,%ymm14
    5d6a:	00 
    5d6b:	c5 05 ef b9 e0 01 00 	vpxor  0x1e0(%rcx),%ymm15,%ymm15
    5d72:	00 
    for (op = 0; operations[op][0] >= 0; op++) {
    5d73:	4d 8b 01             	mov    (%r9),%r8
    5d76:	49 83 c1 08          	add    $0x8,%r9
    5d7a:	49 63 28             	movslq (%r8),%rbp
    5d7d:	85 ed                	test   %ebp,%ebp
    5d7f:	0f 88 56 01 00 00    	js     5edb <jerasure_do_scheduled_operations+0x21b>
      sptr = ptrs[operations[op][0]] + operations[op][1]*packetsize + i;
    5d85:	41 8b 48 04          	mov    0x4(%r8),%ecx
    5d89:	49 89 c4             	mov    %rax,%r12
    5d8c:	0f af ca             	imul   %edx,%ecx
      dptr = ptrs[operations[op][2]] + operations[op][3]*packetsize + i;
    5d8f:	41 8b 40 0c          	mov    0xc(%r8),%eax
    5d93:	0f af c2             	imul   %edx,%eax
      sptr = ptrs[operations[op][0]] + operations[op][1]*packetsize + i;
    5d96:	48 63 c9             	movslq %ecx,%rcx
    5d99:	4c 01 d1             	add    %r10,%rcx
    5d9c:	48 03 0c ef          	add    (%rdi,%rbp,8),%rcx
      dptr = ptrs[operations[op][2]] + operations[op][3]*packetsize + i;
    5da0:	48 98                	cltq   
    5da2:	49 63 68 08          	movslq 0x8(%r8),%rbp
      if (operations[op][4]) {
    5da6:	45 8b 40 10          	mov    0x10(%r8),%r8d
      dptr = ptrs[operations[op][2]] + operations[op][3]*packetsize + i;
    5daa:	4c 01 d0             	add    %r10,%rax
    5dad:	48 03 04 ef          	add    (%rdi,%rbp,8),%rax
      if (operations[op][4]) {
    5db1:	45 85 c0             	test   %r8d,%r8d
    5db4:	0f 85 46 ff ff ff    	jne    5d00 <jerasure_do_scheduled_operations+0x40>
          : "ymm1", "ymm2", "ymm3", "ymm4"
        );
        // jerasure_total_xor_bytes += 512;
      } else {
        // write previous result to memory and copy current data to last 8 ymm registers
        if (prev_dptr != NULL) {
    5dba:	4d 85 e4             	test   %r12,%r12
    5dbd:	0f 84 93 00 00 00    	je     5e56 <jerasure_do_scheduled_operations+0x196>
          __asm__ __volatile__ (
    5dc3:	c4 c1 7d 7f 04 24    	vmovdqa %ymm0,(%r12)
    5dc9:	c4 c1 7d 7f 4c 24 20 	vmovdqa %ymm1,0x20(%r12)
    5dd0:	c4 c1 7d 7f 54 24 40 	vmovdqa %ymm2,0x40(%r12)
    5dd7:	c4 c1 7d 7f 5c 24 60 	vmovdqa %ymm3,0x60(%r12)
    5dde:	c4 c1 7d 7f a4 24 80 	vmovdqa %ymm4,0x80(%r12)
    5de5:	00 00 00 
    5de8:	c4 c1 7d 7f ac 24 a0 	vmovdqa %ymm5,0xa0(%r12)
    5def:	00 00 00 
    5df2:	c4 c1 7d 7f b4 24 c0 	vmovdqa %ymm6,0xc0(%r12)
    5df9:	00 00 00 
    5dfc:	c4 c1 7d 7f bc 24 e0 	vmovdqa %ymm7,0xe0(%r12)
    5e03:	00 00 00 
    5e06:	c4 41 7d 7f 84 24 00 	vmovdqa %ymm8,0x100(%r12)
    5e0d:	01 00 00 
    5e10:	c4 41 7d 7f 8c 24 20 	vmovdqa %ymm9,0x120(%r12)
    5e17:	01 00 00 
    5e1a:	c4 41 7d 7f 94 24 40 	vmovdqa %ymm10,0x140(%r12)
    5e21:	01 00 00 
    5e24:	c4 41 7d 7f 9c 24 60 	vmovdqa %ymm11,0x160(%r12)
    5e2b:	01 00 00 
    5e2e:	c4 41 7d 7f a4 24 80 	vmovdqa %ymm12,0x180(%r12)
    5e35:	01 00 00 
    5e38:	c4 41 7d 7f ac 24 a0 	vmovdqa %ymm13,0x1a0(%r12)
    5e3f:	01 00 00 
    5e42:	c4 41 7d 7f b4 24 c0 	vmovdqa %ymm14,0x1c0(%r12)
    5e49:	01 00 00 
    5e4c:	c4 41 7d 7f bc 24 e0 	vmovdqa %ymm15,0x1e0(%r12)
    5e53:	01 00 00 
            :
            : "r"(sptr), "r"(prev_dptr)
            : "ymm1", "ymm2", "ymm3", "ymm4"
          );
        }
        __asm__ __volatile__ (
    5e56:	c5 fd 6f 01          	vmovdqa (%rcx),%ymm0
    5e5a:	c5 fd 6f 49 20       	vmovdqa 0x20(%rcx),%ymm1
    5e5f:	c5 fd 6f 51 40       	vmovdqa 0x40(%rcx),%ymm2
    5e64:	c5 fd 6f 59 60       	vmovdqa 0x60(%rcx),%ymm3
    5e69:	c5 fd 6f a1 80 00 00 	vmovdqa 0x80(%rcx),%ymm4
    5e70:	00 
    5e71:	c5 fd 6f a9 a0 00 00 	vmovdqa 0xa0(%rcx),%ymm5
    5e78:	00 
    5e79:	c5 fd 6f b1 c0 00 00 	vmovdqa 0xc0(%rcx),%ymm6
    5e80:	00 
    5e81:	c5 fd 6f b9 e0 00 00 	vmovdqa 0xe0(%rcx),%ymm7
    5e88:	00 
    5e89:	c5 7d 6f 81 00 01 00 	vmovdqa 0x100(%rcx),%ymm8
    5e90:	00 
    5e91:	c5 7d 6f 89 20 01 00 	vmovdqa 0x120(%rcx),%ymm9
    5e98:	00 
    5e99:	c5 7d 6f 91 40 01 00 	vmovdqa 0x140(%rcx),%ymm10
    5ea0:	00 
    5ea1:	c5 7d 6f 99 60 01 00 	vmovdqa 0x160(%rcx),%ymm11
    5ea8:	00 
    5ea9:	c5 7d 6f a1 80 01 00 	vmovdqa 0x180(%rcx),%ymm12
    5eb0:	00 
    5eb1:	c5 7d 6f a9 a0 01 00 	vmovdqa 0x1a0(%rcx),%ymm13
    5eb8:	00 
    5eb9:	c5 7d 6f b1 c0 01 00 	vmovdqa 0x1c0(%rcx),%ymm14
    5ec0:	00 
    5ec1:	c5 7d 6f b9 e0 01 00 	vmovdqa 0x1e0(%rcx),%ymm15
    5ec8:	00 
    for (op = 0; operations[op][0] >= 0; op++) {
    5ec9:	4d 8b 01             	mov    (%r9),%r8
    5ecc:	49 83 c1 08          	add    $0x8,%r9
    5ed0:	49 63 28             	movslq (%r8),%rbp
    5ed3:	85 ed                	test   %ebp,%ebp
    5ed5:	0f 89 aa fe ff ff    	jns    5d85 <jerasure_do_scheduled_operations+0xc5>
        // jerasure_total_memcpy_bytes += 512;
      }
      prev_dptr = dptr;
    }
    // don't forget write last result
    __asm__ __volatile__ (
    5edb:	c5 fd 7f 00          	vmovdqa %ymm0,(%rax)
    5edf:	c5 fd 7f 48 20       	vmovdqa %ymm1,0x20(%rax)
    5ee4:	c5 fd 7f 50 40       	vmovdqa %ymm2,0x40(%rax)
    5ee9:	c5 fd 7f 58 60       	vmovdqa %ymm3,0x60(%rax)
    5eee:	c5 fd 7f a0 80 00 00 	vmovdqa %ymm4,0x80(%rax)
    5ef5:	00 
    5ef6:	c5 fd 7f a8 a0 00 00 	vmovdqa %ymm5,0xa0(%rax)
    5efd:	00 
    5efe:	c5 fd 7f b0 c0 00 00 	vmovdqa %ymm6,0xc0(%rax)
    5f05:	00 
    5f06:	c5 fd 7f b8 e0 00 00 	vmovdqa %ymm7,0xe0(%rax)
    5f0d:	00 
    5f0e:	c5 7d 7f 80 00 01 00 	vmovdqa %ymm8,0x100(%rax)
    5f15:	00 
    5f16:	c5 7d 7f 88 20 01 00 	vmovdqa %ymm9,0x120(%rax)
    5f1d:	00 
    5f1e:	c5 7d 7f 90 40 01 00 	vmovdqa %ymm10,0x140(%rax)
    5f25:	00 
    5f26:	c5 7d 7f 98 60 01 00 	vmovdqa %ymm11,0x160(%rax)
    5f2d:	00 
    5f2e:	c5 7d 7f a0 80 01 00 	vmovdqa %ymm12,0x180(%rax)
    5f35:	00 
    5f36:	c5 7d 7f a8 a0 01 00 	vmovdqa %ymm13,0x1a0(%rax)
    5f3d:	00 
    5f3e:	c5 7d 7f b0 c0 01 00 	vmovdqa %ymm14,0x1c0(%rax)
    5f45:	00 
    5f46:	c5 7d 7f b8 e0 01 00 	vmovdqa %ymm15,0x1e0(%rax)
    5f4d:	00 
  for (i = 0; i < packetsize; i += 512)
    5f4e:	49 81 c2 00 02 00 00 	add    $0x200,%r10
    5f55:	44 39 d2             	cmp    %r10d,%edx
    5f58:	0f 8f 82 fd ff ff    	jg     5ce0 <jerasure_do_scheduled_operations+0x20>
    5f5e:	c5 f8 77             	vzeroupper 
      : "r"(sptr), "r"(prev_dptr)
      : "ymm1", "ymm2", "ymm3", "ymm4"
    );
    // jerasure_total_memcpy_bytes += 512;
  }
}
    5f61:	5b                   	pop    %rbx
    5f62:	5d                   	pop    %rbp
    5f63:	41 5c                	pop    %r12
    5f65:	c3                   	retq   
    5f66:	c3                   	retq   
    5f67:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    5f6e:	00 00 

0000000000005f70 <jerasure_schedule_decode_cache>:
{
    5f70:	f3 0f 1e fa          	endbr64 
    5f74:	41 57                	push   %r15
    5f76:	49 89 ca             	mov    %rcx,%r10
    5f79:	4c 89 c9             	mov    %r9,%rcx
    5f7c:	41 56                	push   %r14
    5f7e:	41 55                	push   %r13
    5f80:	41 54                	push   %r12
    5f82:	55                   	push   %rbp
    5f83:	53                   	push   %rbx
    5f84:	89 d3                	mov    %edx,%ebx
    5f86:	4c 89 c2             	mov    %r8,%rdx
    5f89:	48 83 ec 18          	sub    $0x18,%rsp
  if (erasures[1] == -1) {
    5f8d:	45 8b 40 04          	mov    0x4(%r8),%r8d
{
    5f91:	44 8b 6c 24 60       	mov    0x60(%rsp),%r13d
  if (erasures[1] == -1) {
    5f96:	41 83 f8 ff          	cmp    $0xffffffff,%r8d
    5f9a:	0f 84 96 00 00 00    	je     6036 <jerasure_schedule_decode_cache+0xc6>
  } else if (erasures[2] == -1) {
    5fa0:	83 7a 08 ff          	cmpl   $0xffffffff,0x8(%rdx)
    5fa4:	0f 85 a0 00 00 00    	jne    604a <jerasure_schedule_decode_cache+0xda>
    index = erasures[0]*(k+m) + erasures[1];
    5faa:	8b 02                	mov    (%rdx),%eax
    5fac:	8d 2c 37             	lea    (%rdi,%rsi,1),%ebp
    5faf:	0f af c5             	imul   %ebp,%eax
    5fb2:	44 01 c0             	add    %r8d,%eax
  ptrs = set_up_ptrs_for_scheduled_decoding(k, m, erasures, data_ptrs, coding_ptrs);
    5fb5:	4c 8b 44 24 50       	mov    0x50(%rsp),%r8
  schedule = scache[index];
    5fba:	48 98                	cltq   
    5fbc:	4d 8b 34 c2          	mov    (%r10,%rax,8),%r14
  ptrs = set_up_ptrs_for_scheduled_decoding(k, m, erasures, data_ptrs, coding_ptrs);
    5fc0:	e8 9b e5 ff ff       	callq  4560 <set_up_ptrs_for_scheduled_decoding>
    5fc5:	48 89 c7             	mov    %rax,%rdi
  if (ptrs == NULL) return -1;
    5fc8:	48 85 c0             	test   %rax,%rax
    5fcb:	74 7d                	je     604a <jerasure_schedule_decode_cache+0xda>
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    5fcd:	8b 44 24 58          	mov    0x58(%rsp),%eax
    5fd1:	85 c0                	test   %eax,%eax
    5fd3:	7e 4b                	jle    6020 <jerasure_schedule_decode_cache+0xb0>
    for (i = 0; i < k+m; i++) ptrs[i] += (packetsize*w);
    5fd5:	41 0f af dd          	imul   %r13d,%ebx
    5fd9:	8d 45 ff             	lea    -0x1(%rbp),%eax
    5fdc:	4c 8d 7c c7 08       	lea    0x8(%rdi,%rax,8),%r15
    5fe1:	89 5c 24 0c          	mov    %ebx,0xc(%rsp)
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    5fe5:	45 31 e4             	xor    %r12d,%r12d
    for (i = 0; i < k+m; i++) ptrs[i] += (packetsize*w);
    5fe8:	48 63 db             	movslq %ebx,%rbx
    5feb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  jerasure_do_scheduled_operations(ptrs, schedule, packetsize);
    5ff0:	44 89 ea             	mov    %r13d,%edx
    5ff3:	4c 89 f6             	mov    %r14,%rsi
    5ff6:	e8 c5 fc ff ff       	callq  5cc0 <jerasure_do_scheduled_operations>
    for (i = 0; i < k+m; i++) ptrs[i] += (packetsize*w);
    5ffb:	48 89 fa             	mov    %rdi,%rdx
    5ffe:	85 ed                	test   %ebp,%ebp
    6000:	7e 12                	jle    6014 <jerasure_schedule_decode_cache+0xa4>
    6002:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    6008:	48 01 1a             	add    %rbx,(%rdx)
    600b:	48 83 c2 08          	add    $0x8,%rdx
    600f:	49 39 d7             	cmp    %rdx,%r15
    6012:	75 f4                	jne    6008 <jerasure_schedule_decode_cache+0x98>
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    6014:	44 03 64 24 0c       	add    0xc(%rsp),%r12d
    6019:	44 39 64 24 58       	cmp    %r12d,0x58(%rsp)
    601e:	7f d0                	jg     5ff0 <jerasure_schedule_decode_cache+0x80>
  free(ptrs);
    6020:	e8 5b b2 ff ff       	callq  1280 <free@plt>
  return 0;
    6025:	31 c0                	xor    %eax,%eax
}
    6027:	48 83 c4 18          	add    $0x18,%rsp
    602b:	5b                   	pop    %rbx
    602c:	5d                   	pop    %rbp
    602d:	41 5c                	pop    %r12
    602f:	41 5d                	pop    %r13
    6031:	41 5e                	pop    %r14
    6033:	41 5f                	pop    %r15
    6035:	c3                   	retq   
    index = erasures[0]*(k+m) + erasures[0];
    6036:	44 8b 02             	mov    (%rdx),%r8d
    6039:	8d 2c 37             	lea    (%rdi,%rsi,1),%ebp
    603c:	89 e8                	mov    %ebp,%eax
    603e:	41 0f af c0          	imul   %r8d,%eax
    6042:	44 01 c0             	add    %r8d,%eax
    6045:	e9 6b ff ff ff       	jmpq   5fb5 <jerasure_schedule_decode_cache+0x45>
    return -1;
    604a:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    604f:	eb d6                	jmp    6027 <jerasure_schedule_decode_cache+0xb7>
    6051:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    6058:	00 00 00 00 
    605c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000006060 <jerasure_schedule_encode>:

void jerasure_schedule_encode(int k, int m, int w, int **schedule,
                                   char **data_ptrs, char **coding_ptrs, int size, int packetsize)
{
    6060:	f3 0f 1e fa          	endbr64 
    6064:	41 57                	push   %r15
    6066:	4c 63 ff             	movslq %edi,%r15
    6069:	41 56                	push   %r14
    606b:	41 55                	push   %r13
    606d:	41 89 f5             	mov    %esi,%r13d
    6070:	41 54                	push   %r12
  char **ptr_copy;
  int i, j, tdone;

  ptr_copy = talloc(char *, (k+m));
    6072:	45 8d 24 37          	lea    (%r15,%rsi,1),%r12d
    6076:	49 63 fc             	movslq %r12d,%rdi
{
    6079:	55                   	push   %rbp
  ptr_copy = talloc(char *, (k+m));
    607a:	48 c1 e7 03          	shl    $0x3,%rdi
{
    607e:	48 89 cd             	mov    %rcx,%rbp
    6081:	53                   	push   %rbx
    6082:	89 d3                	mov    %edx,%ebx
    6084:	48 83 ec 28          	sub    $0x28,%rsp
    6088:	8b 44 24 60          	mov    0x60(%rsp),%eax
    608c:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
    6091:	4c 89 4c 24 10       	mov    %r9,0x10(%rsp)
    6096:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    609a:	44 8b 74 24 68       	mov    0x68(%rsp),%r14d
  ptr_copy = talloc(char *, (k+m));
    609f:	e8 5c b3 ff ff       	callq  1400 <malloc@plt>
    60a4:	31 d2                	xor    %edx,%edx
  for (i = 0; i < k; i++) ptr_copy[i] = data_ptrs[i];
    60a6:	45 85 ff             	test   %r15d,%r15d
    60a9:	4c 8b 4c 24 10       	mov    0x10(%rsp),%r9
    60ae:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  ptr_copy = talloc(char *, (k+m));
    60b3:	48 89 c7             	mov    %rax,%rdi
  for (i = 0; i < k; i++) ptr_copy[i] = data_ptrs[i];
    60b6:	41 8d 4f ff          	lea    -0x1(%r15),%ecx
    60ba:	7e 17                	jle    60d3 <jerasure_schedule_encode+0x73>
    60bc:	0f 1f 40 00          	nopl   0x0(%rax)
    60c0:	49 8b 04 d0          	mov    (%r8,%rdx,8),%rax
    60c4:	48 89 04 d7          	mov    %rax,(%rdi,%rdx,8)
    60c8:	48 89 d0             	mov    %rdx,%rax
    60cb:	48 ff c2             	inc    %rdx
    60ce:	48 39 c1             	cmp    %rax,%rcx
    60d1:	75 ed                	jne    60c0 <jerasure_schedule_encode+0x60>
  for (i = 0; i < m; i++) ptr_copy[i+k] = coding_ptrs[i];
    60d3:	45 85 ed             	test   %r13d,%r13d
    60d6:	7e 23                	jle    60fb <jerasure_schedule_encode+0x9b>
    60d8:	41 8d 75 ff          	lea    -0x1(%r13),%esi
    60dc:	4a 8d 0c ff          	lea    (%rdi,%r15,8),%rcx
    60e0:	31 d2                	xor    %edx,%edx
    60e2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    60e8:	49 8b 04 d1          	mov    (%r9,%rdx,8),%rax
    60ec:	48 89 04 d1          	mov    %rax,(%rcx,%rdx,8)
    60f0:	48 89 d0             	mov    %rdx,%rax
    60f3:	48 ff c2             	inc    %rdx
    60f6:	48 39 c6             	cmp    %rax,%rsi
    60f9:	75 ed                	jne    60e8 <jerasure_schedule_encode+0x88>
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    60fb:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    60ff:	85 c0                	test   %eax,%eax
    6101:	7e 4d                	jle    6150 <jerasure_schedule_encode+0xf0>
    jerasure_do_scheduled_operations(ptr_copy, schedule, packetsize);
    for (i = 0; i < k+m; i++) ptr_copy[i] += (packetsize*w);
    6103:	41 0f af de          	imul   %r14d,%ebx
    6107:	41 8d 44 24 ff       	lea    -0x1(%r12),%eax
    610c:	4c 8d 7c c7 08       	lea    0x8(%rdi,%rax,8),%r15
    6111:	89 5c 24 10          	mov    %ebx,0x10(%rsp)
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    6115:	45 31 ed             	xor    %r13d,%r13d
    for (i = 0; i < k+m; i++) ptr_copy[i] += (packetsize*w);
    6118:	48 63 db             	movslq %ebx,%rbx
    611b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    jerasure_do_scheduled_operations(ptr_copy, schedule, packetsize);
    6120:	44 89 f2             	mov    %r14d,%edx
    6123:	48 89 ee             	mov    %rbp,%rsi
    6126:	e8 95 fb ff ff       	callq  5cc0 <jerasure_do_scheduled_operations>
    for (i = 0; i < k+m; i++) ptr_copy[i] += (packetsize*w);
    612b:	48 89 fa             	mov    %rdi,%rdx
    612e:	45 85 e4             	test   %r12d,%r12d
    6131:	7e 11                	jle    6144 <jerasure_schedule_encode+0xe4>
    6133:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    6138:	48 01 1a             	add    %rbx,(%rdx)
    613b:	48 83 c2 08          	add    $0x8,%rdx
    613f:	49 39 d7             	cmp    %rdx,%r15
    6142:	75 f4                	jne    6138 <jerasure_schedule_encode+0xd8>
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    6144:	44 03 6c 24 10       	add    0x10(%rsp),%r13d
    6149:	44 39 6c 24 0c       	cmp    %r13d,0xc(%rsp)
    614e:	7f d0                	jg     6120 <jerasure_schedule_encode+0xc0>
  }
  free(ptr_copy);
}
    6150:	48 83 c4 28          	add    $0x28,%rsp
    6154:	5b                   	pop    %rbx
    6155:	5d                   	pop    %rbp
    6156:	41 5c                	pop    %r12
    6158:	41 5d                	pop    %r13
    615a:	41 5e                	pop    %r14
    615c:	41 5f                	pop    %r15
  free(ptr_copy);
    615e:	e9 1d b1 ff ff       	jmpq   1280 <free@plt>
    6163:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    616a:	00 00 00 00 
    616e:	66 90                	xchg   %ax,%ax

0000000000006170 <jerasure_dumb_bitmatrix_to_schedule>:
    
int **jerasure_dumb_bitmatrix_to_schedule(int k, int m, int w, int *bitmatrix)
{
    6170:	f3 0f 1e fa          	endbr64 
    6174:	41 57                	push   %r15
    6176:	41 56                	push   %r14
    6178:	41 89 fe             	mov    %edi,%r14d
    617b:	41 55                	push   %r13
    617d:	41 54                	push   %r12
    617f:	55                   	push   %rbp
    6180:	89 d5                	mov    %edx,%ebp
    6182:	53                   	push   %rbx
    6183:	89 f3                	mov    %esi,%ebx

  operations = talloc(int *, k*m*w*w+1);
  op = 0;
  
  index = 0;
  for (i = 0; i < m*w; i++) {
    6185:	0f af dd             	imul   %ebp,%ebx
{
    6188:	48 83 ec 48          	sub    $0x48,%rsp
    618c:	89 7c 24 20          	mov    %edi,0x20(%rsp)
  operations = talloc(int *, k*m*w*w+1);
    6190:	0f af fe             	imul   %esi,%edi
{
    6193:	89 54 24 24          	mov    %edx,0x24(%rsp)
    6197:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
  operations = talloc(int *, k*m*w*w+1);
    619c:	0f af fa             	imul   %edx,%edi
    619f:	0f af fa             	imul   %edx,%edi
    61a2:	ff c7                	inc    %edi
    61a4:	48 63 ff             	movslq %edi,%rdi
    61a7:	48 c1 e7 03          	shl    $0x3,%rdi
    61ab:	e8 50 b2 ff ff       	callq  1400 <malloc@plt>
    61b0:	49 89 c7             	mov    %rax,%r15
  for (i = 0; i < m*w; i++) {
    61b3:	89 5c 24 38          	mov    %ebx,0x38(%rsp)
    61b7:	85 db                	test   %ebx,%ebx
    61b9:	0f 8e 3a 01 00 00    	jle    62f9 <jerasure_dumb_bitmatrix_to_schedule+0x189>
    optodo = 0;
    for (j = 0; j < k*w; j++) {
    61bf:	44 0f af f5          	imul   %ebp,%r14d
  for (i = 0; i < m*w; i++) {
    61c3:	c7 44 24 10 00 00 00 	movl   $0x0,0x10(%rsp)
    61ca:	00 
  index = 0;
    61cb:	c7 44 24 2c 00 00 00 	movl   $0x0,0x2c(%rsp)
    61d2:	00 
    61d3:	41 8d 46 ff          	lea    -0x1(%r14),%eax
    61d7:	89 44 24 3c          	mov    %eax,0x3c(%rsp)
  op = 0;
    61db:	45 31 ed             	xor    %r13d,%r13d
    for (j = 0; j < k*w; j++) {
    61de:	44 89 74 24 28       	mov    %r14d,0x28(%rsp)
    61e3:	4d 89 fe             	mov    %r15,%r14
    61e6:	45 89 ef             	mov    %r13d,%r15d
    61e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    61f0:	8b 44 24 28          	mov    0x28(%rsp),%eax
    61f4:	85 c0                	test   %eax,%eax
    61f6:	0f 8e f4 00 00 00    	jle    62f0 <jerasure_dumb_bitmatrix_to_schedule+0x180>
    61fc:	44 8b 54 24 3c       	mov    0x3c(%rsp),%r10d
    6201:	48 63 44 24 2c       	movslq 0x2c(%rsp),%rax
    6206:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    optodo = 0;
    620b:	45 31 e4             	xor    %r12d,%r12d
    620e:	4c 8d 0c 81          	lea    (%rcx,%rax,4),%r9
    for (j = 0; j < k*w; j++) {
    6212:	31 db                	xor    %ebx,%ebx
    6214:	4d 89 d5             	mov    %r10,%r13
    6217:	44 89 e2             	mov    %r12d,%edx
    621a:	eb 10                	jmp    622c <jerasure_dumb_bitmatrix_to_schedule+0xbc>
    621c:	0f 1f 40 00          	nopl   0x0(%rax)
    6220:	48 8d 43 01          	lea    0x1(%rbx),%rax
    6224:	49 39 dd             	cmp    %rbx,%r13
    6227:	74 7a                	je     62a3 <jerasure_dumb_bitmatrix_to_schedule+0x133>
    6229:	48 89 c3             	mov    %rax,%rbx
      if (bitmatrix[index]) {
    622c:	49 63 ef             	movslq %r15d,%rbp
    622f:	41 8b 0c 99          	mov    (%r9,%rbx,4),%ecx
    6233:	48 c1 e5 03          	shl    $0x3,%rbp
    6237:	49 8d 34 2e          	lea    (%r14,%rbp,1),%rsi
    623b:	85 c9                	test   %ecx,%ecx
    623d:	74 e1                	je     6220 <jerasure_dumb_bitmatrix_to_schedule+0xb0>
        operations[op] = talloc(int, 5);
    623f:	bf 14 00 00 00       	mov    $0x14,%edi
    6244:	48 89 74 24 18       	mov    %rsi,0x18(%rsp)
    6249:	89 54 24 14          	mov    %edx,0x14(%rsp)
    624d:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
    6252:	e8 a9 b1 ff ff       	callq  1400 <malloc@plt>
    6257:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
        operations[op][4] = optodo;
    625c:	8b 54 24 14          	mov    0x14(%rsp),%edx
        operations[op] = talloc(int, 5);
    6260:	48 89 c1             	mov    %rax,%rcx
        operations[op][4] = optodo;
    6263:	89 50 10             	mov    %edx,0x10(%rax)
        operations[op] = talloc(int, 5);
    6266:	48 89 06             	mov    %rax,(%rsi)
        operations[op][0] = j/w;
    6269:	8b 7c 24 24          	mov    0x24(%rsp),%edi
    626d:	89 d8                	mov    %ebx,%eax
    626f:	99                   	cltd   
    6270:	f7 ff                	idiv   %edi
        operations[op][1] = j%w;
        operations[op][2] = k+i/w;
        operations[op][3] = i%w;
        optodo = 1;
    6272:	4c 8b 4c 24 08       	mov    0x8(%rsp),%r9
        op++;
    6277:	41 ff c7             	inc    %r15d
    627a:	49 8d 74 2e 08       	lea    0x8(%r14,%rbp,1),%rsi
        operations[op][0] = j/w;
    627f:	89 01                	mov    %eax,(%rcx)
        operations[op][1] = j%w;
    6281:	8b 44 24 10          	mov    0x10(%rsp),%eax
    6285:	89 51 04             	mov    %edx,0x4(%rcx)
        operations[op][2] = k+i/w;
    6288:	99                   	cltd   
    6289:	f7 ff                	idiv   %edi
    628b:	03 44 24 20          	add    0x20(%rsp),%eax
    628f:	89 41 08             	mov    %eax,0x8(%rcx)
        operations[op][3] = i%w;
    6292:	89 51 0c             	mov    %edx,0xc(%rcx)
        op++;
    6295:	48 8d 43 01          	lea    0x1(%rbx),%rax
        optodo = 1;
    6299:	ba 01 00 00 00       	mov    $0x1,%edx
    for (j = 0; j < k*w; j++) {
    629e:	49 39 dd             	cmp    %rbx,%r13
    62a1:	75 86                	jne    6229 <jerasure_dumb_bitmatrix_to_schedule+0xb9>
    62a3:	8b 54 24 28          	mov    0x28(%rsp),%edx
    62a7:	49 89 f5             	mov    %rsi,%r13
    62aa:	01 54 24 2c          	add    %edx,0x2c(%rsp)
  for (i = 0; i < m*w; i++) {
    62ae:	ff 44 24 10          	incl   0x10(%rsp)
    62b2:	8b 44 24 10          	mov    0x10(%rsp),%eax
    62b6:	3b 44 24 38          	cmp    0x38(%rsp),%eax
    62ba:	0f 85 30 ff ff ff    	jne    61f0 <jerasure_dumb_bitmatrix_to_schedule+0x80>
    62c0:	4d 89 f7             	mov    %r14,%r15
    62c3:	4d 89 ee             	mov    %r13,%r14
        
      }
      index++;
    }
  }
  operations[op] = talloc(int, 5);
    62c6:	bf 14 00 00 00       	mov    $0x14,%edi
    62cb:	e8 30 b1 ff ff       	callq  1400 <malloc@plt>
    62d0:	49 89 06             	mov    %rax,(%r14)
  operations[op][0] = -1;
    62d3:	c7 00 ff ff ff ff    	movl   $0xffffffff,(%rax)
  return operations;
}
    62d9:	48 83 c4 48          	add    $0x48,%rsp
    62dd:	4c 89 f8             	mov    %r15,%rax
    62e0:	5b                   	pop    %rbx
    62e1:	5d                   	pop    %rbp
    62e2:	41 5c                	pop    %r12
    62e4:	41 5d                	pop    %r13
    62e6:	41 5e                	pop    %r14
    62e8:	41 5f                	pop    %r15
    62ea:	c3                   	retq   
    62eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    62f0:	49 63 c7             	movslq %r15d,%rax
    62f3:	4d 8d 2c c6          	lea    (%r14,%rax,8),%r13
    62f7:	eb b5                	jmp    62ae <jerasure_dumb_bitmatrix_to_schedule+0x13e>
  for (i = 0; i < m*w; i++) {
    62f9:	49 89 c6             	mov    %rax,%r14
    62fc:	eb c8                	jmp    62c6 <jerasure_dumb_bitmatrix_to_schedule+0x156>
    62fe:	66 90                	xchg   %ax,%ax

0000000000006300 <jerasure_smart_bitmatrix_to_schedule>:

int **jerasure_smart_bitmatrix_to_schedule(int k, int m, int w, int *bitmatrix)
{
    6300:	f3 0f 1e fa          	endbr64 
    6304:	41 57                	push   %r15
    6306:	41 56                	push   %r14
    6308:	41 55                	push   %r13
    630a:	41 89 fd             	mov    %edi,%r13d
    630d:	41 54                	push   %r12
    630f:	41 89 d4             	mov    %edx,%r12d
    6312:	55                   	push   %rbp
    6313:	53                   	push   %rbx
    6314:	89 f3                	mov    %esi,%ebx
  jerasure_print_bitmatrix(bitmatrix, m*w, k*w, w); */

  operations = talloc(int *, k*m*w*w+1);
  op = 0;
  
  diff = talloc(int, m*w);
    6316:	41 0f af dc          	imul   %r12d,%ebx
{
    631a:	48 81 ec 88 00 00 00 	sub    $0x88,%rsp
    6321:	89 7c 24 38          	mov    %edi,0x38(%rsp)
  operations = talloc(int *, k*m*w*w+1);
    6325:	0f af fe             	imul   %esi,%edi
  diff = talloc(int, m*w);
    6328:	48 63 eb             	movslq %ebx,%rbp
    632b:	48 c1 e5 02          	shl    $0x2,%rbp
  operations = talloc(int *, k*m*w*w+1);
    632f:	0f af fa             	imul   %edx,%edi
{
    6332:	48 89 4c 24 40       	mov    %rcx,0x40(%rsp)
  operations = talloc(int *, k*m*w*w+1);
    6337:	0f af fa             	imul   %edx,%edi
    633a:	ff c7                	inc    %edi
    633c:	48 63 ff             	movslq %edi,%rdi
    633f:	48 c1 e7 03          	shl    $0x3,%rdi
    6343:	e8 b8 b0 ff ff       	callq  1400 <malloc@plt>
  diff = talloc(int, m*w);
    6348:	48 89 ef             	mov    %rbp,%rdi
  operations = talloc(int *, k*m*w*w+1);
    634b:	49 89 c7             	mov    %rax,%r15
  diff = talloc(int, m*w);
    634e:	e8 ad b0 ff ff       	callq  1400 <malloc@plt>
  from = talloc(int, m*w);
    6353:	48 89 ef             	mov    %rbp,%rdi
  diff = talloc(int, m*w);
    6356:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  from = talloc(int, m*w);
    635b:	e8 a0 b0 ff ff       	callq  1400 <malloc@plt>
  flink = talloc(int, m*w);
    6360:	48 89 ef             	mov    %rbp,%rdi
  from = talloc(int, m*w);
    6363:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  flink = talloc(int, m*w);
    6368:	e8 93 b0 ff ff       	callq  1400 <malloc@plt>
  blink = talloc(int, m*w);
    636d:	48 89 ef             	mov    %rbp,%rdi
  flink = talloc(int, m*w);
    6370:	49 89 c6             	mov    %rax,%r14
  blink = talloc(int, m*w);
    6373:	e8 88 b0 ff ff       	callq  1400 <malloc@plt>

  ptr = bitmatrix;

  bestdiff = k*w+1;
    6378:	45 89 ea             	mov    %r13d,%r10d
    637b:	45 0f af d4          	imul   %r12d,%r10d
  blink = talloc(int, m*w);
    637f:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
  bestdiff = k*w+1;
    6384:	45 8d 4a 01          	lea    0x1(%r10),%r9d
    6388:	44 89 4c 24 70       	mov    %r9d,0x70(%rsp)
  top = 0;
  for (i = 0; i < m*w; i++) {
    638d:	85 db                	test   %ebx,%ebx
    638f:	0f 8e 95 00 00 00    	jle    642a <jerasure_smart_bitmatrix_to_schedule+0x12a>
    6395:	48 89 c7             	mov    %rax,%rdi
    6398:	41 8d 42 ff          	lea    -0x1(%r10),%eax
    639c:	48 89 6c 24 08       	mov    %rbp,0x8(%rsp)
    63a1:	4c 8d 1c 85 04 00 00 	lea    0x4(,%rax,4),%r11
    63a8:	00 
    63a9:	48 8b 6c 24 68       	mov    0x68(%rsp),%rbp
  ptr = bitmatrix;
    63ae:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    63b3:	4c 8b 6c 24 18       	mov    0x18(%rsp),%r13
    63b8:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
    63bd:	ff cb                	dec    %ebx
  for (i = 0; i < m*w; i++) {
    63bf:	31 f6                	xor    %esi,%esi
    63c1:	49 89 ff             	mov    %rdi,%r15
    63c4:	0f 1f 40 00          	nopl   0x0(%rax)
    63c8:	89 f7                	mov    %esi,%edi
    63ca:	41 89 f0             	mov    %esi,%r8d
    no = 0;
    for (j = 0; j < k*w; j++) {
    63cd:	4a 8d 0c 18          	lea    (%rax,%r11,1),%rcx
    no = 0;
    63d1:	31 d2                	xor    %edx,%edx
    for (j = 0; j < k*w; j++) {
    63d3:	45 85 d2             	test   %r10d,%r10d
    63d6:	7e 13                	jle    63eb <jerasure_smart_bitmatrix_to_schedule+0xeb>
    63d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    63df:	00 
      no += *ptr;
    63e0:	03 10                	add    (%rax),%edx
      ptr++;
    63e2:	48 83 c0 04          	add    $0x4,%rax
    for (j = 0; j < k*w; j++) {
    63e6:	48 39 c8             	cmp    %rcx,%rax
    63e9:	75 f5                	jne    63e0 <jerasure_smart_bitmatrix_to_schedule+0xe0>
    }
    diff[i] = no;
    from[i] = -1;
    flink[i] = i+1;
    63eb:	8d 4f 01             	lea    0x1(%rdi),%ecx
    blink[i] = i-1;
    63ee:	ff cf                	dec    %edi
    diff[i] = no;
    63f0:	89 54 b5 00          	mov    %edx,0x0(%rbp,%rsi,4)
    from[i] = -1;
    63f4:	41 c7 44 b5 00 ff ff 	movl   $0xffffffff,0x0(%r13,%rsi,4)
    63fb:	ff ff 
    flink[i] = i+1;
    63fd:	41 89 0c b6          	mov    %ecx,(%r14,%rsi,4)
    blink[i] = i-1;
    6401:	41 89 3c b7          	mov    %edi,(%r15,%rsi,4)
    if (no < bestdiff) {
    6405:	41 39 d1             	cmp    %edx,%r9d
    6408:	7e 08                	jle    6412 <jerasure_smart_bitmatrix_to_schedule+0x112>
    640a:	44 89 44 24 14       	mov    %r8d,0x14(%rsp)
    640f:	41 89 d1             	mov    %edx,%r9d
  for (i = 0; i < m*w; i++) {
    6412:	48 8d 56 01          	lea    0x1(%rsi),%rdx
    6416:	48 39 f3             	cmp    %rsi,%rbx
    6419:	74 05                	je     6420 <jerasure_smart_bitmatrix_to_schedule+0x120>
    641b:	48 89 d6             	mov    %rdx,%rsi
    641e:	eb a8                	jmp    63c8 <jerasure_smart_bitmatrix_to_schedule+0xc8>
    6420:	48 8b 6c 24 08       	mov    0x8(%rsp),%rbp
    6425:	4c 8b 7c 24 20       	mov    0x20(%rsp),%r15
      bestdiff = no;
      bestrow = i;
    }
  }

  flink[m*w-1] = -1;
    642a:	41 8d 42 ff          	lea    -0x1(%r10),%eax
    642e:	41 c7 44 2e fc ff ff 	movl   $0xffffffff,-0x4(%r14,%rbp,1)
    6435:	ff ff 
  top = 0;
    6437:	c7 44 24 5c 00 00 00 	movl   $0x0,0x5c(%rsp)
    643e:	00 
    643f:	89 44 24 58          	mov    %eax,0x58(%rsp)
  op = 0;
    6443:	45 31 c9             	xor    %r9d,%r9d
    6446:	4c 89 7c 24 60       	mov    %r15,0x60(%rsp)
    644b:	45 89 cf             	mov    %r9d,%r15d
    644e:	4d 89 f1             	mov    %r14,%r9
    6451:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  
  while (top != -1) {
    row = bestrow;
    /* printf("Doing row %d - %d from %d\n", row, diff[row], from[row]);  */

    if (blink[row] == -1) {
    6458:	48 63 54 24 14       	movslq 0x14(%rsp),%rdx
    645d:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    6462:	8b 0c 90             	mov    (%rax,%rdx,4),%ecx
    6465:	49 63 04 91          	movslq (%r9,%rdx,4),%rax
    6469:	83 f9 ff             	cmp    $0xffffffff,%ecx
    646c:	0f 84 92 03 00 00    	je     6804 <jerasure_smart_bitmatrix_to_schedule+0x504>
      top = flink[row];
      if (top != -1) blink[top] = -1;
    } else {
      flink[blink[row]] = flink[row];
    6472:	48 63 f9             	movslq %ecx,%rdi
    6475:	41 89 04 b9          	mov    %eax,(%r9,%rdi,4)
      if (flink[row] != -1) {
    6479:	83 f8 ff             	cmp    $0xffffffff,%eax
    647c:	74 08                	je     6486 <jerasure_smart_bitmatrix_to_schedule+0x186>
        blink[flink[row]] = blink[row];
    647e:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
    6483:	89 0c 86             	mov    %ecx,(%rsi,%rax,4)
      }
    }

    ptr = bitmatrix + row*k*w;
    6486:	8b 44 24 38          	mov    0x38(%rsp),%eax
    648a:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
    648f:	0f af 44 24 14       	imul   0x14(%rsp),%eax
    6494:	4d 63 ef             	movslq %r15d,%r13
    6497:	4e 8d 34 ed 00 00 00 	lea    0x0(,%r13,8),%r14
    649e:	00 
    649f:	41 0f af c4          	imul   %r12d,%eax
    64a3:	48 98                	cltq   
    64a5:	48 8d 1c 86          	lea    (%rsi,%rax,4),%rbx
    if (from[row] == -1) {
    64a9:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    64ae:	48 8b 74 24 60       	mov    0x60(%rsp),%rsi
    64b3:	8b 2c 90             	mov    (%rax,%rdx,4),%ebp
    64b6:	4e 8d 04 36          	lea    (%rsi,%r14,1),%r8
    64ba:	83 fd ff             	cmp    $0xffffffff,%ebp
    64bd:	0f 85 9c 01 00 00    	jne    665f <jerasure_smart_bitmatrix_to_schedule+0x35f>
      optodo = 0;
      for (j = 0; j < k*w; j++) {
    64c3:	45 85 d2             	test   %r10d,%r10d
    64c6:	0f 8e cc 00 00 00    	jle    6598 <jerasure_smart_bitmatrix_to_schedule+0x298>
      optodo = 0;
    64cc:	31 d2                	xor    %edx,%edx
    64ce:	4c 89 4c 24 48       	mov    %r9,0x48(%rsp)
    64d3:	44 89 54 24 3c       	mov    %r10d,0x3c(%rsp)
    64d8:	45 89 e1             	mov    %r12d,%r9d
    64db:	44 8b 5c 24 58       	mov    0x58(%rsp),%r11d
    64e0:	49 89 dc             	mov    %rbx,%r12
      for (j = 0; j < k*w; j++) {
    64e3:	31 ed                	xor    %ebp,%ebp
    64e5:	45 89 fe             	mov    %r15d,%r14d
    64e8:	89 d3                	mov    %edx,%ebx
    64ea:	eb 17                	jmp    6503 <jerasure_smart_bitmatrix_to_schedule+0x203>
    64ec:	0f 1f 40 00          	nopl   0x0(%rax)
    64f0:	48 8d 45 01          	lea    0x1(%rbp),%rax
    64f4:	49 39 eb             	cmp    %rbp,%r11
    64f7:	0f 84 88 00 00 00    	je     6585 <jerasure_smart_bitmatrix_to_schedule+0x285>
    64fd:	48 89 c5             	mov    %rax,%rbp
    6500:	4d 63 ee             	movslq %r14d,%r13
        if (ptr[j]) {
    6503:	41 8b 04 ac          	mov    (%r12,%rbp,4),%eax
    6507:	49 c1 e5 03          	shl    $0x3,%r13
    650b:	4e 8d 04 2e          	lea    (%rsi,%r13,1),%r8
    650f:	85 c0                	test   %eax,%eax
    6511:	74 dd                	je     64f0 <jerasure_smart_bitmatrix_to_schedule+0x1f0>
          operations[op] = talloc(int, 5);
    6513:	bf 14 00 00 00       	mov    $0x14,%edi
    6518:	4c 89 44 24 30       	mov    %r8,0x30(%rsp)
    651d:	44 89 4c 24 28       	mov    %r9d,0x28(%rsp)
    6522:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
    6527:	4c 89 5c 24 08       	mov    %r11,0x8(%rsp)
    652c:	e8 cf ae ff ff       	callq  1400 <malloc@plt>
    6531:	4c 8b 44 24 30       	mov    0x30(%rsp),%r8
          operations[op][4] = optodo;
    6536:	89 58 10             	mov    %ebx,0x10(%rax)
          operations[op] = talloc(int, 5);
    6539:	48 89 c7             	mov    %rax,%rdi
    653c:	49 89 00             	mov    %rax,(%r8)
          operations[op][0] = j/w;
    653f:	44 8b 4c 24 28       	mov    0x28(%rsp),%r9d
    6544:	89 e8                	mov    %ebp,%eax
    6546:	99                   	cltd   
    6547:	41 f7 f9             	idiv   %r9d
    654a:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
          operations[op][1] = j%w;
          operations[op][2] = k+row/w;
          operations[op][3] = row%w;
          optodo = 1;
    654f:	4c 8b 5c 24 08       	mov    0x8(%rsp),%r11
          op++;
    6554:	41 ff c6             	inc    %r14d
    6557:	4e 8d 44 2e 08       	lea    0x8(%rsi,%r13,1),%r8
          optodo = 1;
    655c:	bb 01 00 00 00       	mov    $0x1,%ebx
          operations[op][0] = j/w;
    6561:	89 07                	mov    %eax,(%rdi)
          operations[op][1] = j%w;
    6563:	8b 44 24 14          	mov    0x14(%rsp),%eax
    6567:	89 57 04             	mov    %edx,0x4(%rdi)
          operations[op][2] = k+row/w;
    656a:	99                   	cltd   
    656b:	41 f7 f9             	idiv   %r9d
    656e:	03 44 24 38          	add    0x38(%rsp),%eax
    6572:	89 47 08             	mov    %eax,0x8(%rdi)
          operations[op][3] = row%w;
    6575:	89 57 0c             	mov    %edx,0xc(%rdi)
      for (j = 0; j < k*w; j++) {
    6578:	48 8d 45 01          	lea    0x1(%rbp),%rax
    657c:	49 39 eb             	cmp    %rbp,%r11
    657f:	0f 85 78 ff ff ff    	jne    64fd <jerasure_smart_bitmatrix_to_schedule+0x1fd>
    6585:	4c 89 e3             	mov    %r12,%rbx
    6588:	44 8b 54 24 3c       	mov    0x3c(%rsp),%r10d
    658d:	45 89 cc             	mov    %r9d,%r12d
    6590:	4c 8b 4c 24 48       	mov    0x48(%rsp),%r9
    6595:	45 89 f7             	mov    %r14d,%r15d
          op++;
        }
      }
    }
    bestdiff = k*w+1;
    for (i = top; i != -1; i = flink[i]) {
    6598:	44 8b 5c 24 5c       	mov    0x5c(%rsp),%r11d
    659d:	41 83 fb ff          	cmp    $0xffffffff,%r11d
    65a1:	0f 84 fd 01 00 00    	je     67a4 <jerasure_smart_bitmatrix_to_schedule+0x4a4>
    65a7:	44 8b 44 24 58       	mov    0x58(%rsp),%r8d
    65ac:	44 89 7c 24 20       	mov    %r15d,0x20(%rsp)
    65b1:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
    bestdiff = k*w+1;
    65b6:	8b 6c 24 70          	mov    0x70(%rsp),%ebp
    for (i = top; i != -1; i = flink[i]) {
    65ba:	44 8b 6c 24 14       	mov    0x14(%rsp),%r13d
    65bf:	4c 8b 74 24 68       	mov    0x68(%rsp),%r14
    65c4:	8b 74 24 38          	mov    0x38(%rsp),%esi
    65c8:	4c 8b 7c 24 40       	mov    0x40(%rsp),%r15
    65cd:	0f 1f 00             	nopl   (%rax)
      no = 1;
      b1 = bitmatrix + i*k*w;
    65d0:	89 f0                	mov    %esi,%eax
    65d2:	41 0f af c3          	imul   %r11d,%eax
    65d6:	41 0f af c4          	imul   %r12d,%eax
    65da:	48 98                	cltq   
      for (j = 0; j < k*w; j++) no += (ptr[j] ^ b1[j]);
    65dc:	45 85 d2             	test   %r10d,%r10d
    65df:	7e 77                	jle    6658 <jerasure_smart_bitmatrix_to_schedule+0x358>
      no = 1;
    65e1:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
    65e6:	49 8d 3c 87          	lea    (%r15,%rax,4),%rdi
    65ea:	b9 01 00 00 00       	mov    $0x1,%ecx
      for (j = 0; j < k*w; j++) no += (ptr[j] ^ b1[j]);
    65ef:	31 c0                	xor    %eax,%eax
    65f1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    65f8:	8b 14 83             	mov    (%rbx,%rax,4),%edx
    65fb:	33 14 87             	xor    (%rdi,%rax,4),%edx
    65fe:	01 d1                	add    %edx,%ecx
    6600:	48 89 c2             	mov    %rax,%rdx
    6603:	48 ff c0             	inc    %rax
    6606:	49 39 d0             	cmp    %rdx,%r8
    6609:	75 ed                	jne    65f8 <jerasure_smart_bitmatrix_to_schedule+0x2f8>
    660b:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
      if (no < diff[i]) {
    6610:	49 63 d3             	movslq %r11d,%rdx
    6613:	49 8d 3c 96          	lea    (%r14,%rdx,4),%rdi
    6617:	8b 07                	mov    (%rdi),%eax
    6619:	39 c8                	cmp    %ecx,%eax
    661b:	7e 12                	jle    662f <jerasure_smart_bitmatrix_to_schedule+0x32f>
        from[i] = row;
    661d:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    6622:	44 8b 44 24 14       	mov    0x14(%rsp),%r8d
        diff[i] = no;
    6627:	89 0f                	mov    %ecx,(%rdi)
        from[i] = row;
    6629:	44 89 04 90          	mov    %r8d,(%rax,%rdx,4)
        diff[i] = no;
    662d:	89 c8                	mov    %ecx,%eax
      }
      if (diff[i] < bestdiff) {
    662f:	39 c5                	cmp    %eax,%ebp
    6631:	7e 05                	jle    6638 <jerasure_smart_bitmatrix_to_schedule+0x338>
    6633:	89 c5                	mov    %eax,%ebp
    6635:	45 89 dd             	mov    %r11d,%r13d
    for (i = top; i != -1; i = flink[i]) {
    6638:	45 8b 1c 91          	mov    (%r9,%rdx,4),%r11d
    663c:	41 83 fb ff          	cmp    $0xffffffff,%r11d
    6640:	75 8e                	jne    65d0 <jerasure_smart_bitmatrix_to_schedule+0x2d0>
    6642:	44 89 6c 24 14       	mov    %r13d,0x14(%rsp)
    6647:	44 8b 7c 24 20       	mov    0x20(%rsp),%r15d
    664c:	e9 07 fe ff ff       	jmpq   6458 <jerasure_smart_bitmatrix_to_schedule+0x158>
    6651:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
      no = 1;
    6658:	b9 01 00 00 00       	mov    $0x1,%ecx
    665d:	eb b1                	jmp    6610 <jerasure_smart_bitmatrix_to_schedule+0x310>
      operations[op] = talloc(int, 5);
    665f:	bf 14 00 00 00       	mov    $0x14,%edi
    6664:	4c 89 44 24 28       	mov    %r8,0x28(%rsp)
    6669:	4c 89 4c 24 20       	mov    %r9,0x20(%rsp)
    666e:	44 89 54 24 08       	mov    %r10d,0x8(%rsp)
    6673:	e8 88 ad ff ff       	callq  1400 <malloc@plt>
    6678:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
      operations[op][4] = 0;
    667d:	c7 40 10 00 00 00 00 	movl   $0x0,0x10(%rax)
      operations[op] = talloc(int, 5);
    6684:	48 89 c1             	mov    %rax,%rcx
    6687:	49 89 00             	mov    %rax,(%r8)
      operations[op][0] = k+from[row]/w;
    668a:	89 e8                	mov    %ebp,%eax
    668c:	99                   	cltd   
    668d:	41 f7 fc             	idiv   %r12d
    6690:	8b 74 24 38          	mov    0x38(%rsp),%esi
      for (j = 0; j < k*w; j++) {
    6694:	44 8b 54 24 08       	mov    0x8(%rsp),%r10d
    6699:	4c 8b 4c 24 20       	mov    0x20(%rsp),%r9
      b1 = bitmatrix + from[row]*k*w;
    669e:	0f af ee             	imul   %esi,%ebp
      op++;
    66a1:	41 ff c7             	inc    %r15d
      b1 = bitmatrix + from[row]*k*w;
    66a4:	41 0f af ec          	imul   %r12d,%ebp
    66a8:	48 63 ed             	movslq %ebp,%rbp
      operations[op][0] = k+from[row]/w;
    66ab:	01 f0                	add    %esi,%eax
    66ad:	89 01                	mov    %eax,(%rcx)
      operations[op][1] = from[row]%w;
    66af:	8b 44 24 14          	mov    0x14(%rsp),%eax
    66b3:	89 51 04             	mov    %edx,0x4(%rcx)
      operations[op][2] = k+row/w;
    66b6:	99                   	cltd   
    66b7:	41 f7 fc             	idiv   %r12d
    66ba:	01 f0                	add    %esi,%eax
      for (j = 0; j < k*w; j++) {
    66bc:	45 85 d2             	test   %r10d,%r10d
    66bf:	89 54 24 48          	mov    %edx,0x48(%rsp)
      operations[op][2] = k+row/w;
    66c3:	89 44 24 3c          	mov    %eax,0x3c(%rsp)
    66c7:	89 41 08             	mov    %eax,0x8(%rcx)
      operations[op][3] = row%w;
    66ca:	89 51 0c             	mov    %edx,0xc(%rcx)
      for (j = 0; j < k*w; j++) {
    66cd:	0f 8e 5a 01 00 00    	jle    682d <jerasure_smart_bitmatrix_to_schedule+0x52d>
    66d3:	8b 44 24 58          	mov    0x58(%rsp),%eax
    66d7:	44 89 64 24 30       	mov    %r12d,0x30(%rsp)
    66dc:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    66e1:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    66e6:	44 89 54 24 74       	mov    %r10d,0x74(%rsp)
    66eb:	4c 8d 34 a8          	lea    (%rax,%rbp,4),%r14
    66ef:	48 89 d8             	mov    %rbx,%rax
    66f2:	4d 89 f4             	mov    %r14,%r12
    66f5:	44 89 fb             	mov    %r15d,%ebx
    66f8:	4c 89 4c 24 78       	mov    %r9,0x78(%rsp)
    66fd:	4c 8b 74 24 60       	mov    0x60(%rsp),%r14
    6702:	31 ed                	xor    %ebp,%ebp
    6704:	49 89 c7             	mov    %rax,%r15
    6707:	eb 0a                	jmp    6713 <jerasure_smart_bitmatrix_to_schedule+0x413>
    6709:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    6710:	48 89 c5             	mov    %rax,%rbp
    6713:	4c 63 eb             	movslq %ebx,%r13
        if (ptr[j] ^ b1[j]) {
    6716:	41 8b 04 ac          	mov    (%r12,%rbp,4),%eax
    671a:	49 c1 e5 03          	shl    $0x3,%r13
    671e:	89 6c 24 08          	mov    %ebp,0x8(%rsp)
    6722:	4f 8d 04 2e          	lea    (%r14,%r13,1),%r8
    6726:	41 39 04 af          	cmp    %eax,(%r15,%rbp,4)
    672a:	74 46                	je     6772 <jerasure_smart_bitmatrix_to_schedule+0x472>
          operations[op] = talloc(int, 5);
    672c:	bf 14 00 00 00       	mov    $0x14,%edi
    6731:	4c 89 44 24 28       	mov    %r8,0x28(%rsp)
    6736:	e8 c5 ac ff ff       	callq  1400 <malloc@plt>
    673b:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
    6740:	8b 54 24 08          	mov    0x8(%rsp),%edx
          operations[op][4] = 1;
    6744:	c7 40 10 01 00 00 00 	movl   $0x1,0x10(%rax)
          operations[op] = talloc(int, 5);
    674b:	48 89 c1             	mov    %rax,%rcx
    674e:	49 89 00             	mov    %rax,(%r8)
          operations[op][0] = j/w;
    6751:	89 d0                	mov    %edx,%eax
    6753:	99                   	cltd   
    6754:	f7 7c 24 30          	idivl  0x30(%rsp)
          op++;
    6758:	ff c3                	inc    %ebx
    675a:	4f 8d 44 2e 08       	lea    0x8(%r14,%r13,1),%r8
          operations[op][0] = j/w;
    675f:	89 01                	mov    %eax,(%rcx)
          operations[op][2] = k+row/w;
    6761:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
          operations[op][1] = j%w;
    6765:	89 51 04             	mov    %edx,0x4(%rcx)
          operations[op][2] = k+row/w;
    6768:	89 41 08             	mov    %eax,0x8(%rcx)
          operations[op][3] = row%w;
    676b:	8b 44 24 48          	mov    0x48(%rsp),%eax
    676f:	89 41 0c             	mov    %eax,0xc(%rcx)
      for (j = 0; j < k*w; j++) {
    6772:	48 8d 45 01          	lea    0x1(%rbp),%rax
    6776:	48 39 6c 24 20       	cmp    %rbp,0x20(%rsp)
    677b:	75 93                	jne    6710 <jerasure_smart_bitmatrix_to_schedule+0x410>
    for (i = top; i != -1; i = flink[i]) {
    677d:	44 8b 5c 24 5c       	mov    0x5c(%rsp),%r11d
    6782:	4c 89 f8             	mov    %r15,%rax
    6785:	44 8b 54 24 74       	mov    0x74(%rsp),%r10d
    678a:	41 89 df             	mov    %ebx,%r15d
    678d:	4c 8b 4c 24 78       	mov    0x78(%rsp),%r9
    6792:	44 8b 64 24 30       	mov    0x30(%rsp),%r12d
    6797:	48 89 c3             	mov    %rax,%rbx
    679a:	41 83 fb ff          	cmp    $0xffffffff,%r11d
    679e:	0f 85 03 fe ff ff    	jne    65a7 <jerasure_smart_bitmatrix_to_schedule+0x2a7>
        bestrow = i;
      }
    }
  }
  
  operations[op] = talloc(int, 5);
    67a4:	bf 14 00 00 00       	mov    $0x14,%edi
    67a9:	4d 89 ce             	mov    %r9,%r14
    67ac:	4c 8b 7c 24 60       	mov    0x60(%rsp),%r15
    67b1:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
    67b6:	e8 45 ac ff ff       	callq  1400 <malloc@plt>
    67bb:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
  operations[op][0] = -1;
    67c0:	c7 00 ff ff ff ff    	movl   $0xffffffff,(%rax)
  operations[op] = talloc(int, 5);
    67c6:	49 89 00             	mov    %rax,(%r8)
  free(from);
    67c9:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    67ce:	e8 ad aa ff ff       	callq  1280 <free@plt>
  free(diff);
    67d3:	48 8b 7c 24 68       	mov    0x68(%rsp),%rdi
    67d8:	e8 a3 aa ff ff       	callq  1280 <free@plt>
  free(blink);
    67dd:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
    67e2:	e8 99 aa ff ff       	callq  1280 <free@plt>
  free(flink);
    67e7:	4c 89 f7             	mov    %r14,%rdi
    67ea:	e8 91 aa ff ff       	callq  1280 <free@plt>

  return operations;
}
    67ef:	48 81 c4 88 00 00 00 	add    $0x88,%rsp
    67f6:	5b                   	pop    %rbx
    67f7:	5d                   	pop    %rbp
    67f8:	41 5c                	pop    %r12
    67fa:	41 5d                	pop    %r13
    67fc:	41 5e                	pop    %r14
    67fe:	4c 89 f8             	mov    %r15,%rax
    6801:	41 5f                	pop    %r15
    6803:	c3                   	retq   
      if (top != -1) blink[top] = -1;
    6804:	c7 44 24 5c ff ff ff 	movl   $0xffffffff,0x5c(%rsp)
    680b:	ff 
    680c:	83 f8 ff             	cmp    $0xffffffff,%eax
    680f:	0f 84 71 fc ff ff    	je     6486 <jerasure_smart_bitmatrix_to_schedule+0x186>
    6815:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
    681a:	48 63 c8             	movslq %eax,%rcx
    681d:	c7 04 8e ff ff ff ff 	movl   $0xffffffff,(%rsi,%rcx,4)
    6824:	89 44 24 5c          	mov    %eax,0x5c(%rsp)
    6828:	e9 59 fc ff ff       	jmpq   6486 <jerasure_smart_bitmatrix_to_schedule+0x186>
    682d:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    6832:	4e 8d 44 30 08       	lea    0x8(%rax,%r14,1),%r8
    6837:	e9 5c fd ff ff       	jmpq   6598 <jerasure_smart_bitmatrix_to_schedule+0x298>
    683c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000006840 <jerasure_generate_decoding_schedule>:
{
    6840:	41 57                	push   %r15
    6842:	41 56                	push   %r14
    6844:	4d 89 c6             	mov    %r8,%r14
    6847:	41 55                	push   %r13
    6849:	41 54                	push   %r12
    684b:	41 89 f4             	mov    %esi,%r12d
    684e:	55                   	push   %rbp
    684f:	53                   	push   %rbx
    6850:	89 d3                	mov    %edx,%ebx
    6852:	48 81 ec b8 00 00 00 	sub    $0xb8,%rsp
  for (i = 0; erasures[i] != -1; i++) {
    6859:	41 8b 00             	mov    (%r8),%eax
{
    685c:	89 7c 24 24          	mov    %edi,0x24(%rsp)
    6860:	48 89 4c 24 78       	mov    %rcx,0x78(%rsp)
    6865:	44 89 8c 24 ac 00 00 	mov    %r9d,0xac(%rsp)
    686c:	00 
  for (i = 0; erasures[i] != -1; i++) {
    686d:	83 f8 ff             	cmp    $0xffffffff,%eax
    6870:	0f 84 8c 06 00 00    	je     6f02 <jerasure_generate_decoding_schedule+0x6c2>
  cdf = 0;
    6876:	c7 44 24 08 00 00 00 	movl   $0x0,0x8(%rsp)
    687d:	00 
    687e:	49 8d 50 04          	lea    0x4(%r8),%rdx
  ddf = 0;
    6882:	31 ed                	xor    %ebp,%ebp
    6884:	eb 0d                	jmp    6893 <jerasure_generate_decoding_schedule+0x53>
  for (i = 0; erasures[i] != -1; i++) {
    6886:	8b 02                	mov    (%rdx),%eax
    6888:	48 83 c2 04          	add    $0x4,%rdx
    if (erasures[i] < k) ddf++; else cdf++;
    688c:	ff c5                	inc    %ebp
  for (i = 0; erasures[i] != -1; i++) {
    688e:	83 f8 ff             	cmp    $0xffffffff,%eax
    6891:	74 15                	je     68a8 <jerasure_generate_decoding_schedule+0x68>
    if (erasures[i] < k) ddf++; else cdf++;
    6893:	39 44 24 24          	cmp    %eax,0x24(%rsp)
    6897:	7f ed                	jg     6886 <jerasure_generate_decoding_schedule+0x46>
  for (i = 0; erasures[i] != -1; i++) {
    6899:	8b 02                	mov    (%rdx),%eax
    689b:	48 83 c2 04          	add    $0x4,%rdx
    if (erasures[i] < k) ddf++; else cdf++;
    689f:	ff 44 24 08          	incl   0x8(%rsp)
  for (i = 0; erasures[i] != -1; i++) {
    68a3:	83 f8 ff             	cmp    $0xffffffff,%eax
    68a6:	75 eb                	jne    6893 <jerasure_generate_decoding_schedule+0x53>
  row_ids = talloc(int, k+m);
    68a8:	8b 7c 24 24          	mov    0x24(%rsp),%edi
    68ac:	46 8d 2c 27          	lea    (%rdi,%r12,1),%r13d
    68b0:	4d 63 fd             	movslq %r13d,%r15
    68b3:	49 c1 e7 02          	shl    $0x2,%r15
    68b7:	4c 89 ff             	mov    %r15,%rdi
    68ba:	e8 41 ab ff ff       	callq  1400 <malloc@plt>
  ind_to_row = talloc(int, k+m);
    68bf:	4c 89 ff             	mov    %r15,%rdi
  row_ids = talloc(int, k+m);
    68c2:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  ind_to_row = talloc(int, k+m);
    68c7:	e8 34 ab ff ff       	callq  1400 <malloc@plt>
  erased = jerasure_erasures_to_erased(k, m, erasures);
    68cc:	4c 89 f2             	mov    %r14,%rdx
    68cf:	44 8b 74 24 24       	mov    0x24(%rsp),%r14d
    68d4:	44 89 e6             	mov    %r12d,%esi
    68d7:	44 89 f7             	mov    %r14d,%edi
  ind_to_row = talloc(int, k+m);
    68da:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    68df:	49 89 c7             	mov    %rax,%r15
  erased = jerasure_erasures_to_erased(k, m, erasures);
    68e2:	e8 e9 db ff ff       	callq  44d0 <jerasure_erasures_to_erased>
    68e7:	49 89 c4             	mov    %rax,%r12
  if (erased == NULL) return -1;
    68ea:	48 85 c0             	test   %rax,%rax
    68ed:	0f 84 45 04 00 00    	je     6d38 <jerasure_generate_decoding_schedule+0x4f8>
  for (i = 0; i < k; i++) {
    68f3:	45 85 f6             	test   %r14d,%r14d
    68f6:	0f 8e fd 05 00 00    	jle    6ef9 <jerasure_generate_decoding_schedule+0x6b9>
    68fc:	4c 8b 4c 24 10       	mov    0x10(%rsp),%r9
    6901:	44 89 f6             	mov    %r14d,%esi
    6904:	45 8d 56 ff          	lea    -0x1(%r14),%r10d
    6908:	44 89 f1             	mov    %r14d,%ecx
    690b:	31 d2                	xor    %edx,%edx
    690d:	4d 89 fb             	mov    %r15,%r11
    6910:	eb 14                	jmp    6926 <jerasure_generate_decoding_schedule+0xe6>
      row_ids[i] = i;
    6912:	41 89 14 91          	mov    %edx,(%r9,%rdx,4)
      ind_to_row[i] = i;
    6916:	41 89 14 93          	mov    %edx,(%r11,%rdx,4)
  for (i = 0; i < k; i++) {
    691a:	48 8d 42 01          	lea    0x1(%rdx),%rax
    691e:	4c 39 d2             	cmp    %r10,%rdx
    6921:	74 5e                	je     6981 <jerasure_generate_decoding_schedule+0x141>
    6923:	48 89 c2             	mov    %rax,%rdx
    if (erased[i] == 0) {
    6926:	41 8b 04 94          	mov    (%r12,%rdx,4),%eax
    692a:	41 89 d0             	mov    %edx,%r8d
    692d:	85 c0                	test   %eax,%eax
    692f:	74 e1                	je     6912 <jerasure_generate_decoding_schedule+0xd2>
      while (erased[j]) j++;
    6931:	4c 63 f1             	movslq %ecx,%r14
    6934:	47 8b 3c b4          	mov    (%r12,%r14,4),%r15d
    6938:	8d 41 01             	lea    0x1(%rcx),%eax
    693b:	4a 8d 3c b5 00 00 00 	lea    0x0(,%r14,4),%rdi
    6942:	00 
    6943:	48 98                	cltq   
    6945:	45 85 ff             	test   %r15d,%r15d
    6948:	74 17                	je     6961 <jerasure_generate_decoding_schedule+0x121>
    694a:	89 c1                	mov    %eax,%ecx
    694c:	48 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%rdi
    6953:	00 
    6954:	48 ff c0             	inc    %rax
    6957:	45 8b 74 84 fc       	mov    -0x4(%r12,%rax,4),%r14d
    695c:	45 85 f6             	test   %r14d,%r14d
    695f:	75 e9                	jne    694a <jerasure_generate_decoding_schedule+0x10a>
      row_ids[x] = i;
    6961:	48 63 c6             	movslq %esi,%rax
      row_ids[i] = j;
    6964:	41 89 0c 91          	mov    %ecx,(%r9,%rdx,4)
      ind_to_row[j] = i;
    6968:	45 89 04 3b          	mov    %r8d,(%r11,%rdi,1)
      row_ids[x] = i;
    696c:	45 89 04 81          	mov    %r8d,(%r9,%rax,4)
      ind_to_row[i] = x;
    6970:	41 89 34 93          	mov    %esi,(%r11,%rdx,4)
      j++;
    6974:	ff c1                	inc    %ecx
      x++;
    6976:	ff c6                	inc    %esi
  for (i = 0; i < k; i++) {
    6978:	48 8d 42 01          	lea    0x1(%rdx),%rax
    697c:	4c 39 d2             	cmp    %r10,%rdx
    697f:	75 a2                	jne    6923 <jerasure_generate_decoding_schedule+0xe3>
  for (i = k; i < k+m; i++) {
    6981:	48 63 44 24 24       	movslq 0x24(%rsp),%rax
    6986:	41 39 c5             	cmp    %eax,%r13d
    6989:	7e 25                	jle    69b0 <jerasure_generate_decoding_schedule+0x170>
    if (erased[i]) {
    698b:	41 8b 3c 84          	mov    (%r12,%rax,4),%edi
    698f:	85 ff                	test   %edi,%edi
    6991:	74 15                	je     69a8 <jerasure_generate_decoding_schedule+0x168>
      row_ids[x] = i;
    6993:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    6998:	48 63 d6             	movslq %esi,%rdx
    699b:	89 04 97             	mov    %eax,(%rdi,%rdx,4)
      ind_to_row[i] = x;
    699e:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    69a3:	89 34 87             	mov    %esi,(%rdi,%rax,4)
      x++;
    69a6:	ff c6                	inc    %esi
  for (i = k; i < k+m; i++) {
    69a8:	48 ff c0             	inc    %rax
    69ab:	41 39 c5             	cmp    %eax,%r13d
    69ae:	7f db                	jg     698b <jerasure_generate_decoding_schedule+0x14b>
  free(erased);
    69b0:	4c 89 e7             	mov    %r12,%rdi
    69b3:	e8 c8 a8 ff ff       	callq  1280 <free@plt>
  real_decoding_matrix = talloc(int, k*w*(cdf+ddf)*w);
    69b8:	44 8b 74 24 24       	mov    0x24(%rsp),%r14d
    69bd:	8b 44 24 08          	mov    0x8(%rsp),%eax
    69c1:	44 0f af f3          	imul   %ebx,%r14d
    69c5:	01 e8                	add    %ebp,%eax
    69c7:	89 84 24 a8 00 00 00 	mov    %eax,0xa8(%rsp)
    69ce:	41 0f af c6          	imul   %r14d,%eax
    69d2:	89 c7                	mov    %eax,%edi
    69d4:	0f af fb             	imul   %ebx,%edi
    69d7:	48 63 ff             	movslq %edi,%rdi
    69da:	48 c1 e7 02          	shl    $0x2,%rdi
    69de:	e8 1d aa ff ff       	callq  1400 <malloc@plt>
    69e3:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  if (ddf > 0) {
    69e8:	85 ed                	test   %ebp,%ebp
    69ea:	0f 85 79 03 00 00    	jne    6d69 <jerasure_generate_decoding_schedule+0x529>
  for (x = 0; x < cdf; x++) {
    69f0:	8b 4c 24 08          	mov    0x8(%rsp),%ecx
    69f4:	85 c9                	test   %ecx,%ecx
    69f6:	0f 84 f9 02 00 00    	je     6cf5 <jerasure_generate_decoding_schedule+0x4b5>
    memcpy(ptr, bitmatrix+drive*k*w*w, sizeof(int)*k*w*w);
    69fc:	48 63 f3             	movslq %ebx,%rsi
    69ff:	48 89 f7             	mov    %rsi,%rdi
    6a02:	48 0f af fe          	imul   %rsi,%rdi
    6a06:	48 63 54 24 24       	movslq 0x24(%rsp),%rdx
    ptr = real_decoding_matrix + k*w*w*(ddf+x);
    6a0b:	44 89 f0             	mov    %r14d,%eax
    memcpy(ptr, bitmatrix+drive*k*w*w, sizeof(int)*k*w*w);
    6a0e:	48 0f af fa          	imul   %rdx,%rdi
    ptr = real_decoding_matrix + k*w*w*(ddf+x);
    6a12:	0f af c3             	imul   %ebx,%eax
    memcpy(ptr, bitmatrix+drive*k*w*w, sizeof(int)*k*w*w);
    6a15:	49 89 d3             	mov    %rdx,%r11
    6a18:	48 8d 3c bd 00 00 00 	lea    0x0(,%rdi,4),%rdi
    6a1f:	00 
    6a20:	48 89 bc 24 80 00 00 	mov    %rdi,0x80(%rsp)
    6a27:	00 
          bzero(ptr+j*k*w+i*w, sizeof(int)*w);
    6a28:	48 8d 3c b5 00 00 00 	lea    0x0(,%rsi,4),%rdi
    6a2f:	00 
    6a30:	48 63 d5             	movslq %ebp,%rdx
    6a33:	48 89 bc 24 88 00 00 	mov    %rdi,0x88(%rsp)
    6a3a:	00 
    6a3b:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    6a40:	4c 01 da             	add    %r11,%rdx
    6a43:	48 8d 34 97          	lea    (%rdi,%rdx,4),%rsi
    6a47:	48 63 c8             	movslq %eax,%rcx
    6a4a:	0f af c5             	imul   %ebp,%eax
    6a4d:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
    6a52:	48 8d 34 8d 00 00 00 	lea    0x0(,%rcx,4),%rsi
    6a59:	00 
    6a5a:	48 89 b4 24 a0 00 00 	mov    %rsi,0xa0(%rsp)
    6a61:	00 
    6a62:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
    6a67:	48 98                	cltq   
    6a69:	48 8d 04 86          	lea    (%rsi,%rax,4),%rax
    6a6d:	48 89 c6             	mov    %rax,%rsi
    6a70:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    6a75:	41 8d 46 ff          	lea    -0x1(%r14),%eax
    6a79:	48 8d 44 86 04       	lea    0x4(%rsi,%rax,4),%rax
    6a7e:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    6a83:	8b 44 24 08          	mov    0x8(%rsp),%eax
    6a87:	44 0f af db          	imul   %ebx,%r11d
    6a8b:	ff c8                	dec    %eax
    6a8d:	48 01 c2             	add    %rax,%rdx
    6a90:	48 8d 44 97 04       	lea    0x4(%rdi,%rdx,4),%rax
    6a95:	48 89 84 24 98 00 00 	mov    %rax,0x98(%rsp)
    6a9c:	00 
    6a9d:	49 63 c3             	movslq %r11d,%rax
    6aa0:	48 89 c5             	mov    %rax,%rbp
    6aa3:	4c 8d 2c 85 00 00 00 	lea    0x0(,%rax,4),%r13
    6aaa:	00 
    6aab:	8d 43 ff             	lea    -0x1(%rbx),%eax
    6aae:	48 89 84 24 90 00 00 	mov    %rax,0x90(%rsp)
    6ab5:	00 
    6ab6:	48 f7 d0             	not    %rax
    6ab9:	48 c1 e0 02          	shl    $0x2,%rax
    6abd:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
    drive = row_ids[x+ddf+k]-k;
    6ac2:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
    6ac7:	44 8b 7c 24 24       	mov    0x24(%rsp),%r15d
    6acc:	8b 00                	mov    (%rax),%eax
    memcpy(ptr, bitmatrix+drive*k*w*w, sizeof(int)*k*w*w);
    6ace:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
    drive = row_ids[x+ddf+k]-k;
    6ad3:	44 29 f8             	sub    %r15d,%eax
    memcpy(ptr, bitmatrix+drive*k*w*w, sizeof(int)*k*w*w);
    6ad6:	41 0f af c7          	imul   %r15d,%eax
    6ada:	48 8b 94 24 80 00 00 	mov    0x80(%rsp),%rdx
    6ae1:	00 
    6ae2:	0f af c3             	imul   %ebx,%eax
    6ae5:	0f af c3             	imul   %ebx,%eax
    6ae8:	48 98                	cltq   
    6aea:	48 8d 34 87          	lea    (%rdi,%rax,4),%rsi
    6aee:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    6af3:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    6af8:	e8 c3 a8 ff ff       	callq  13c0 <memcpy@plt>
    for (i = 0; i < k; i++) {
    6afd:	45 85 ff             	test   %r15d,%r15d
    6b00:	0f 8e c4 01 00 00    	jle    6cca <jerasure_generate_decoding_schedule+0x48a>
    6b06:	8b 44 24 24          	mov    0x24(%rsp),%eax
    6b0a:	31 ff                	xor    %edi,%edi
    6b0c:	ff c8                	dec    %eax
    6b0e:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    6b13:	31 c0                	xor    %eax,%eax
    6b15:	44 89 74 24 28       	mov    %r14d,0x28(%rsp)
    6b1a:	89 6c 24 38          	mov    %ebp,0x38(%rsp)
    6b1e:	4c 8b a4 24 88 00 00 	mov    0x88(%rsp),%r12
    6b25:	00 
    6b26:	41 89 fe             	mov    %edi,%r14d
    6b29:	48 89 c5             	mov    %rax,%rbp
    6b2c:	eb 11                	jmp    6b3f <jerasure_generate_decoding_schedule+0x2ff>
    6b2e:	48 8d 45 01          	lea    0x1(%rbp),%rax
    6b32:	41 01 de             	add    %ebx,%r14d
    6b35:	48 39 6c 24 18       	cmp    %rbp,0x18(%rsp)
    6b3a:	74 4a                	je     6b86 <jerasure_generate_decoding_schedule+0x346>
    6b3c:	48 89 c5             	mov    %rax,%rbp
      if (row_ids[i] != i) {
    6b3f:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    6b44:	39 2c a8             	cmp    %ebp,(%rax,%rbp,4)
    6b47:	74 e5                	je     6b2e <jerasure_generate_decoding_schedule+0x2ee>
        for (j = 0; j < w; j++) {
    6b49:	85 db                	test   %ebx,%ebx
    6b4b:	7e e1                	jle    6b2e <jerasure_generate_decoding_schedule+0x2ee>
    6b4d:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
          bzero(ptr+j*k*w+i*w, sizeof(int)*w);
    6b52:	49 63 c6             	movslq %r14d,%rax
    6b55:	48 8d 3c 87          	lea    (%rdi,%rax,4),%rdi
        for (j = 0; j < w; j++) {
    6b59:	45 31 ff             	xor    %r15d,%r15d
    6b5c:	0f 1f 40 00          	nopl   0x0(%rax)
    6b60:	4c 89 e2             	mov    %r12,%rdx
    6b63:	31 f6                	xor    %esi,%esi
    6b65:	e8 e6 a7 ff ff       	callq  1350 <memset@plt>
    6b6a:	48 89 c7             	mov    %rax,%rdi
    6b6d:	41 ff c7             	inc    %r15d
    6b70:	4c 01 ef             	add    %r13,%rdi
    6b73:	44 39 fb             	cmp    %r15d,%ebx
    6b76:	75 e8                	jne    6b60 <jerasure_generate_decoding_schedule+0x320>
    for (i = 0; i < k; i++) {
    6b78:	48 8d 45 01          	lea    0x1(%rbp),%rax
    6b7c:	41 01 de             	add    %ebx,%r14d
    6b7f:	48 39 6c 24 18       	cmp    %rbp,0x18(%rsp)
    6b84:	75 b6                	jne    6b3c <jerasure_generate_decoding_schedule+0x2fc>
    6b86:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    6b8b:	44 8b 74 24 28       	mov    0x28(%rsp),%r14d
    6b90:	48 03 84 24 90 00 00 	add    0x90(%rsp),%rax
    6b97:	00 
    6b98:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    6b9d:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    6ba2:	c7 44 24 28 00 00 00 	movl   $0x0,0x28(%rsp)
    6ba9:	00 
    6baa:	48 83 c0 04          	add    $0x4,%rax
    6bae:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    6bb3:	8b 6c 24 38          	mov    0x38(%rsp),%ebp
    6bb7:	4c 8b 64 24 48       	mov    0x48(%rsp),%r12
    6bbc:	45 31 ff             	xor    %r15d,%r15d
    6bbf:	eb 16                	jmp    6bd7 <jerasure_generate_decoding_schedule+0x397>
    for (i = 0; i < k; i++) {
    6bc1:	01 5c 24 28          	add    %ebx,0x28(%rsp)
    6bc5:	49 8d 47 01          	lea    0x1(%r15),%rax
    6bc9:	4c 39 7c 24 18       	cmp    %r15,0x18(%rsp)
    6bce:	0f 84 f6 00 00 00    	je     6cca <jerasure_generate_decoding_schedule+0x48a>
    6bd4:	49 89 c7             	mov    %rax,%r15
      if (row_ids[i] != i) {
    6bd7:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    6bdc:	46 39 3c b8          	cmp    %r15d,(%rax,%r15,4)
    6be0:	74 df                	je     6bc1 <jerasure_generate_decoding_schedule+0x381>
        b1 = real_decoding_matrix+(ind_to_row[i]-k)*k*w*w;
    6be2:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    6be7:	42 8b 04 b8          	mov    (%rax,%r15,4),%eax
    6beb:	41 89 c2             	mov    %eax,%r10d
    6bee:	89 44 24 08          	mov    %eax,0x8(%rsp)
    6bf2:	8b 44 24 24          	mov    0x24(%rsp),%eax
    6bf6:	41 29 c2             	sub    %eax,%r10d
    6bf9:	44 0f af d0          	imul   %eax,%r10d
    6bfd:	44 0f af d3          	imul   %ebx,%r10d
    6c01:	44 0f af d3          	imul   %ebx,%r10d
    6c05:	4d 63 d2             	movslq %r10d,%r10
        for (j = 0; j < w; j++) {
    6c08:	85 db                	test   %ebx,%ebx
    6c0a:	7e b5                	jle    6bc1 <jerasure_generate_decoding_schedule+0x381>
    6c0c:	48 63 44 24 28       	movslq 0x28(%rsp),%rax
    6c11:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
    6c16:	4c 89 7c 24 38       	mov    %r15,0x38(%rsp)
    6c1b:	48 03 44 24 58       	add    0x58(%rsp),%rax
            if (bitmatrix[index+j*k*w+i*w+y]) {
    6c20:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
    6c25:	4c 8b 4c 24 30       	mov    0x30(%rsp),%r9
        for (j = 0; j < w; j++) {
    6c2a:	4c 8b 7c 24 68       	mov    0x68(%rsp),%r15
    6c2f:	4c 8d 04 87          	lea    (%rdi,%rax,4),%r8
    6c33:	45 31 db             	xor    %r11d,%r11d
    6c36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    6c3d:	00 00 00 
          for (y = 0; y < w; y++) {
    6c40:	4b 8d 0c 07          	lea    (%r15,%r8,1),%rcx
        for (j = 0; j < w; j++) {
    6c44:	31 ff                	xor    %edi,%edi
    6c46:	eb 13                	jmp    6c5b <jerasure_generate_decoding_schedule+0x41b>
    6c48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    6c4f:	00 
          for (y = 0; y < w; y++) {
    6c50:	48 83 c1 04          	add    $0x4,%rcx
    6c54:	01 ef                	add    %ebp,%edi
    6c56:	49 39 c8             	cmp    %rcx,%r8
    6c59:	74 46                	je     6ca1 <jerasure_generate_decoding_schedule+0x461>
            if (bitmatrix[index+j*k*w+i*w+y]) {
    6c5b:	8b 01                	mov    (%rcx),%eax
    6c5d:	85 c0                	test   %eax,%eax
    6c5f:	74 ef                	je     6c50 <jerasure_generate_decoding_schedule+0x410>
              for (z = 0; z < k*w; z++) {
    6c61:	45 85 f6             	test   %r14d,%r14d
    6c64:	7e ea                	jle    6c50 <jerasure_generate_decoding_schedule+0x410>
    6c66:	48 63 c7             	movslq %edi,%rax
                b2[z] = b2[z] ^ b1[z+y*k*w];
    6c69:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
    6c6e:	4c 01 d0             	add    %r10,%rax
    6c71:	49 8d 14 84          	lea    (%r12,%rax,4),%rdx
    6c75:	4c 89 c8             	mov    %r9,%rax
    6c78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    6c7f:	00 
    6c80:	8b 0a                	mov    (%rdx),%ecx
    6c82:	48 83 c2 04          	add    $0x4,%rdx
    6c86:	31 08                	xor    %ecx,(%rax)
              for (z = 0; z < k*w; z++) {
    6c88:	48 83 c0 04          	add    $0x4,%rax
    6c8c:	48 39 c6             	cmp    %rax,%rsi
    6c8f:	75 ef                	jne    6c80 <jerasure_generate_decoding_schedule+0x440>
    6c91:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
          for (y = 0; y < w; y++) {
    6c96:	01 ef                	add    %ebp,%edi
    6c98:	48 83 c1 04          	add    $0x4,%rcx
    6c9c:	49 39 c8             	cmp    %rcx,%r8
    6c9f:	75 ba                	jne    6c5b <jerasure_generate_decoding_schedule+0x41b>
        for (j = 0; j < w; j++) {
    6ca1:	41 ff c3             	inc    %r11d
    6ca4:	4d 01 e8             	add    %r13,%r8
    6ca7:	4d 01 e9             	add    %r13,%r9
    6caa:	4c 01 ee             	add    %r13,%rsi
    6cad:	44 39 db             	cmp    %r11d,%ebx
    6cb0:	75 8e                	jne    6c40 <jerasure_generate_decoding_schedule+0x400>
    6cb2:	4c 8b 7c 24 38       	mov    0x38(%rsp),%r15
    for (i = 0; i < k; i++) {
    6cb7:	01 5c 24 28          	add    %ebx,0x28(%rsp)
    6cbb:	49 8d 47 01          	lea    0x1(%r15),%rax
    6cbf:	4c 39 7c 24 18       	cmp    %r15,0x18(%rsp)
    6cc4:	0f 85 0a ff ff ff    	jne    6bd4 <jerasure_generate_decoding_schedule+0x394>
  for (x = 0; x < cdf; x++) {
    6cca:	48 83 44 24 70 04    	addq   $0x4,0x70(%rsp)
    6cd0:	48 8b b4 24 a0 00 00 	mov    0xa0(%rsp),%rsi
    6cd7:	00 
    6cd8:	48 01 74 24 30       	add    %rsi,0x30(%rsp)
    6cdd:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
    6ce2:	48 01 74 24 50       	add    %rsi,0x50(%rsp)
    6ce7:	48 39 84 24 98 00 00 	cmp    %rax,0x98(%rsp)
    6cee:	00 
    6cef:	0f 85 cd fd ff ff    	jne    6ac2 <jerasure_generate_decoding_schedule+0x282>
  if (smart) {
    6cf5:	8b 94 24 ac 00 00 00 	mov    0xac(%rsp),%edx
    6cfc:	85 d2                	test   %edx,%edx
    6cfe:	75 4d                	jne    6d4d <jerasure_generate_decoding_schedule+0x50d>
    schedule = jerasure_dumb_bitmatrix_to_schedule(k, ddf+cdf, w, real_decoding_matrix);
    6d00:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    6d05:	8b b4 24 a8 00 00 00 	mov    0xa8(%rsp),%esi
    6d0c:	8b 7c 24 24          	mov    0x24(%rsp),%edi
    6d10:	89 da                	mov    %ebx,%edx
    6d12:	e8 59 f4 ff ff       	callq  6170 <jerasure_dumb_bitmatrix_to_schedule>
    6d17:	49 89 c4             	mov    %rax,%r12
  free(row_ids);
    6d1a:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    6d1f:	e8 5c a5 ff ff       	callq  1280 <free@plt>
  free(ind_to_row);
    6d24:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    6d29:	e8 52 a5 ff ff       	callq  1280 <free@plt>
  free(real_decoding_matrix);
    6d2e:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
    6d33:	e8 48 a5 ff ff       	callq  1280 <free@plt>
}
    6d38:	48 81 c4 b8 00 00 00 	add    $0xb8,%rsp
    6d3f:	5b                   	pop    %rbx
    6d40:	5d                   	pop    %rbp
    6d41:	4c 89 e0             	mov    %r12,%rax
    6d44:	41 5c                	pop    %r12
    6d46:	41 5d                	pop    %r13
    6d48:	41 5e                	pop    %r14
    6d4a:	41 5f                	pop    %r15
    6d4c:	c3                   	retq   
    schedule = jerasure_smart_bitmatrix_to_schedule(k, ddf+cdf, w, real_decoding_matrix);
    6d4d:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    6d52:	8b b4 24 a8 00 00 00 	mov    0xa8(%rsp),%esi
    6d59:	8b 7c 24 24          	mov    0x24(%rsp),%edi
    6d5d:	89 da                	mov    %ebx,%edx
    6d5f:	e8 9c f5 ff ff       	callq  6300 <jerasure_smart_bitmatrix_to_schedule>
    6d64:	49 89 c4             	mov    %rax,%r12
    6d67:	eb b1                	jmp    6d1a <jerasure_generate_decoding_schedule+0x4da>
    decoding_matrix = talloc(int, k*k*w*w);
    6d69:	44 8b 7c 24 24       	mov    0x24(%rsp),%r15d
    6d6e:	44 89 f8             	mov    %r15d,%eax
    6d71:	0f af c3             	imul   %ebx,%eax
    6d74:	0f af c0             	imul   %eax,%eax
    6d77:	4c 63 e8             	movslq %eax,%r13
    6d7a:	49 c1 e5 02          	shl    $0x2,%r13
    6d7e:	4c 89 ef             	mov    %r13,%rdi
    6d81:	e8 7a a6 ff ff       	callq  1400 <malloc@plt>
    6d86:	49 89 c4             	mov    %rax,%r12
    for (i = 0; i < k; i++) {
    6d89:	45 85 ff             	test   %r15d,%r15d
    6d8c:	0f 8e b8 00 00 00    	jle    6e4a <jerasure_generate_decoding_schedule+0x60a>
        memcpy(ptr, bitmatrix+k*w*w*(row_ids[i]-k), k*w*w*sizeof(int));
    6d92:	44 89 f0             	mov    %r14d,%eax
    6d95:	0f af c3             	imul   %ebx,%eax
    6d98:	89 6c 24 50          	mov    %ebp,0x50(%rsp)
    6d9c:	c7 44 24 18 00 00 00 	movl   $0x0,0x18(%rsp)
    6da3:	00 
    6da4:	48 63 d0             	movslq %eax,%rdx
    6da7:	89 44 24 28          	mov    %eax,0x28(%rsp)
    6dab:	8b 44 24 24          	mov    0x24(%rsp),%eax
    6daf:	48 c1 e2 02          	shl    $0x2,%rdx
    6db3:	ff c8                	dec    %eax
    6db5:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    6dba:	49 63 c6             	movslq %r14d,%rax
    6dbd:	48 8d 04 85 04 00 00 	lea    0x4(,%rax,4),%rax
    6dc4:	00 
    6dc5:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    ptr = decoding_matrix;
    6dca:	4c 89 e7             	mov    %r12,%rdi
        memcpy(ptr, bitmatrix+k*w*w*(row_ids[i]-k), k*w*w*sizeof(int));
    6dcd:	45 31 ff             	xor    %r15d,%r15d
    6dd0:	48 89 d5             	mov    %rdx,%rbp
    6dd3:	eb 34                	jmp    6e09 <jerasure_generate_decoding_schedule+0x5c9>
    6dd5:	2b 44 24 24          	sub    0x24(%rsp),%eax
    6dd9:	0f af 44 24 28       	imul   0x28(%rsp),%eax
    6dde:	48 8b 74 24 78       	mov    0x78(%rsp),%rsi
    6de3:	48 89 ea             	mov    %rbp,%rdx
    6de6:	48 98                	cltq   
    6de8:	48 8d 34 86          	lea    (%rsi,%rax,4),%rsi
    6dec:	e8 cf a5 ff ff       	callq  13c0 <memcpy@plt>
    6df1:	48 89 c7             	mov    %rax,%rdi
      ptr += (k*w*w);
    6df4:	01 5c 24 18          	add    %ebx,0x18(%rsp)
    6df8:	48 01 ef             	add    %rbp,%rdi
    for (i = 0; i < k; i++) {
    6dfb:	49 8d 47 01          	lea    0x1(%r15),%rax
    6dff:	4c 3b 7c 24 38       	cmp    0x38(%rsp),%r15
    6e04:	74 40                	je     6e46 <jerasure_generate_decoding_schedule+0x606>
    6e06:	49 89 c7             	mov    %rax,%r15
      if (row_ids[i] == i) {
    6e09:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    6e0e:	42 8b 04 b8          	mov    (%rax,%r15,4),%eax
    6e12:	44 39 f8             	cmp    %r15d,%eax
    6e15:	75 be                	jne    6dd5 <jerasure_generate_decoding_schedule+0x595>
    6e17:	48 89 ea             	mov    %rbp,%rdx
    6e1a:	31 f6                	xor    %esi,%esi
    6e1c:	e8 2f a5 ff ff       	callq  1350 <memset@plt>
    6e21:	48 89 c7             	mov    %rax,%rdi
        for (x = 0; x < w; x++) {
    6e24:	85 db                	test   %ebx,%ebx
    6e26:	7e cc                	jle    6df4 <jerasure_generate_decoding_schedule+0x5b4>
    6e28:	48 63 44 24 18       	movslq 0x18(%rsp),%rax
    6e2d:	48 8d 0c 87          	lea    (%rdi,%rax,4),%rcx
    6e31:	31 c0                	xor    %eax,%eax
    6e33:	ff c0                	inc    %eax
          ptr[x+i*w+x*k*w] = 1;
    6e35:	c7 01 01 00 00 00    	movl   $0x1,(%rcx)
        for (x = 0; x < w; x++) {
    6e3b:	48 03 4c 24 30       	add    0x30(%rsp),%rcx
    6e40:	39 c3                	cmp    %eax,%ebx
    6e42:	75 ef                	jne    6e33 <jerasure_generate_decoding_schedule+0x5f3>
    6e44:	eb ae                	jmp    6df4 <jerasure_generate_decoding_schedule+0x5b4>
    6e46:	8b 6c 24 50          	mov    0x50(%rsp),%ebp
    inverse = talloc(int, k*k*w*w);
    6e4a:	4c 89 ef             	mov    %r13,%rdi
    6e4d:	e8 ae a5 ff ff       	callq  1400 <malloc@plt>
    jerasure_invert_bitmatrix(decoding_matrix, inverse, k*w);
    6e52:	48 89 c6             	mov    %rax,%rsi
    6e55:	44 89 f2             	mov    %r14d,%edx
    6e58:	4c 89 e7             	mov    %r12,%rdi
    6e5b:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    6e60:	e8 5b e1 ff ff       	callq  4fc0 <jerasure_invert_bitmatrix>
    free(decoding_matrix);
    6e65:	4c 89 e7             	mov    %r12,%rdi
    6e68:	e8 13 a4 ff ff       	callq  1280 <free@plt>
      memcpy(ptr, inverse+k*w*w*row_ids[k+i], sizeof(int)*k*w*w);
    6e6d:	48 63 c3             	movslq %ebx,%rax
    6e70:	48 0f af c0          	imul   %rax,%rax
    6e74:	48 63 74 24 24       	movslq 0x24(%rsp),%rsi
    6e79:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    6e7e:	48 0f af c6          	imul   %rsi,%rax
    6e82:	45 89 f4             	mov    %r14d,%r12d
    6e85:	4c 8d 3c b7          	lea    (%rdi,%rsi,4),%r15
    6e89:	48 89 c2             	mov    %rax,%rdx
    6e8c:	8d 45 ff             	lea    -0x1(%rbp),%eax
    6e8f:	44 0f af e3          	imul   %ebx,%r12d
    6e93:	48 01 c6             	add    %rax,%rsi
    6e96:	48 8d 44 b7 04       	lea    0x4(%rdi,%rsi,4),%rax
    6e9b:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
      ptr += (k*w*w);
    6ea0:	4d 63 ec             	movslq %r12d,%r13
    ptr = real_decoding_matrix;
    6ea3:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
    6ea8:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
      memcpy(ptr, inverse+k*w*w*row_ids[k+i], sizeof(int)*k*w*w);
    6ead:	48 c1 e2 02          	shl    $0x2,%rdx
      ptr += (k*w*w);
    6eb1:	49 c1 e5 02          	shl    $0x2,%r13
      memcpy(ptr, inverse+k*w*w*row_ids[k+i], sizeof(int)*k*w*w);
    6eb5:	41 8b 07             	mov    (%r15),%eax
    6eb8:	4c 89 44 24 30       	mov    %r8,0x30(%rsp)
    6ebd:	41 0f af c4          	imul   %r12d,%eax
    6ec1:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
    6ec6:	49 83 c7 04          	add    $0x4,%r15
    6eca:	48 98                	cltq   
    6ecc:	49 8d 34 80          	lea    (%r8,%rax,4),%rsi
    6ed0:	e8 eb a4 ff ff       	callq  13c0 <memcpy@plt>
    6ed5:	48 89 c7             	mov    %rax,%rdi
      ptr += (k*w*w);
    6ed8:	4c 01 ef             	add    %r13,%rdi
    for (i = 0; i < ddf; i++) {
    6edb:	4c 39 7c 24 18       	cmp    %r15,0x18(%rsp)
    6ee0:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    6ee5:	4c 8b 44 24 30       	mov    0x30(%rsp),%r8
    6eea:	75 c9                	jne    6eb5 <jerasure_generate_decoding_schedule+0x675>
    free(inverse);
    6eec:	4c 89 c7             	mov    %r8,%rdi
    6eef:	e8 8c a3 ff ff       	callq  1280 <free@plt>
    6ef4:	e9 f7 fa ff ff       	jmpq   69f0 <jerasure_generate_decoding_schedule+0x1b0>
  for (i = 0; i < k; i++) {
    6ef9:	8b 74 24 24          	mov    0x24(%rsp),%esi
    6efd:	e9 7f fa ff ff       	jmpq   6981 <jerasure_generate_decoding_schedule+0x141>
  cdf = 0;
    6f02:	c7 44 24 08 00 00 00 	movl   $0x0,0x8(%rsp)
    6f09:	00 
  ddf = 0;
    6f0a:	31 ed                	xor    %ebp,%ebp
    6f0c:	e9 97 f9 ff ff       	jmpq   68a8 <jerasure_generate_decoding_schedule+0x68>
    6f11:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    6f18:	00 00 00 00 
    6f1c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000006f20 <jerasure_schedule_decode_lazy>:
{
    6f20:	f3 0f 1e fa          	endbr64 
    6f24:	41 57                	push   %r15
    6f26:	4d 89 c7             	mov    %r8,%r15
    6f29:	41 56                	push   %r14
    6f2b:	41 55                	push   %r13
    6f2d:	41 54                	push   %r12
    6f2f:	41 89 d4             	mov    %edx,%r12d
  ptrs = set_up_ptrs_for_scheduled_decoding(k, m, erasures, data_ptrs, coding_ptrs);
    6f32:	4c 89 fa             	mov    %r15,%rdx
{
    6f35:	55                   	push   %rbp
    6f36:	89 fd                	mov    %edi,%ebp
    6f38:	53                   	push   %rbx
    6f39:	89 f3                	mov    %esi,%ebx
    6f3b:	48 83 ec 18          	sub    $0x18,%rsp
  ptrs = set_up_ptrs_for_scheduled_decoding(k, m, erasures, data_ptrs, coding_ptrs);
    6f3f:	4c 8b 44 24 50       	mov    0x50(%rsp),%r8
{
    6f44:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
    6f49:	4c 89 c9             	mov    %r9,%rcx
    6f4c:	44 8b 74 24 60       	mov    0x60(%rsp),%r14d
  ptrs = set_up_ptrs_for_scheduled_decoding(k, m, erasures, data_ptrs, coding_ptrs);
    6f51:	e8 0a d6 ff ff       	callq  4560 <set_up_ptrs_for_scheduled_decoding>
  if (ptrs == NULL) return -1;
    6f56:	48 85 c0             	test   %rax,%rax
    6f59:	0f 84 a2 00 00 00    	je     7001 <jerasure_schedule_decode_lazy+0xe1>
  schedule = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    6f5f:	4c 8b 5c 24 08       	mov    0x8(%rsp),%r11
    6f64:	44 8b 4c 24 68       	mov    0x68(%rsp),%r9d
    6f69:	89 de                	mov    %ebx,%esi
    6f6b:	4d 89 f8             	mov    %r15,%r8
    6f6e:	4c 89 d9             	mov    %r11,%rcx
    6f71:	44 89 e2             	mov    %r12d,%edx
    6f74:	89 ef                	mov    %ebp,%edi
    6f76:	49 89 c5             	mov    %rax,%r13
    6f79:	e8 c2 f8 ff ff       	callq  6840 <jerasure_generate_decoding_schedule>
    6f7e:	48 89 c6             	mov    %rax,%rsi
  if (schedule == NULL) {
    6f81:	48 85 c0             	test   %rax,%rax
    6f84:	0f 84 7e 00 00 00    	je     7008 <jerasure_schedule_decode_lazy+0xe8>
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    6f8a:	8b 44 24 58          	mov    0x58(%rsp),%eax
    6f8e:	85 c0                	test   %eax,%eax
    6f90:	7e 4e                	jle    6fe0 <jerasure_schedule_decode_lazy+0xc0>
    for (i = 0; i < k+m; i++) ptrs[i] += (packetsize*w);
    6f92:	45 0f af e6          	imul   %r14d,%r12d
    6f96:	01 dd                	add    %ebx,%ebp
    6f98:	8d 45 ff             	lea    -0x1(%rbp),%eax
    6f9b:	44 89 64 24 08       	mov    %r12d,0x8(%rsp)
    6fa0:	4d 63 fc             	movslq %r12d,%r15
    6fa3:	49 8d 5c c5 08       	lea    0x8(%r13,%rax,8),%rbx
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    6fa8:	45 31 e4             	xor    %r12d,%r12d
    6fab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  jerasure_do_scheduled_operations(ptrs, schedule, packetsize);
    6fb0:	44 89 f2             	mov    %r14d,%edx
    6fb3:	4c 89 ef             	mov    %r13,%rdi
    6fb6:	e8 05 ed ff ff       	callq  5cc0 <jerasure_do_scheduled_operations>
    for (i = 0; i < k+m; i++) ptrs[i] += (packetsize*w);
    6fbb:	4c 89 ea             	mov    %r13,%rdx
    6fbe:	85 ed                	test   %ebp,%ebp
    6fc0:	7e 12                	jle    6fd4 <jerasure_schedule_decode_lazy+0xb4>
    6fc2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    6fc8:	4c 01 3a             	add    %r15,(%rdx)
    6fcb:	48 83 c2 08          	add    $0x8,%rdx
    6fcf:	48 39 da             	cmp    %rbx,%rdx
    6fd2:	75 f4                	jne    6fc8 <jerasure_schedule_decode_lazy+0xa8>
  for (tdone = 0; tdone < size; tdone += packetsize*w) {
    6fd4:	44 03 64 24 08       	add    0x8(%rsp),%r12d
    6fd9:	44 39 64 24 58       	cmp    %r12d,0x58(%rsp)
    6fde:	7f d0                	jg     6fb0 <jerasure_schedule_decode_lazy+0x90>
  jerasure_free_schedule(schedule);
    6fe0:	48 89 f7             	mov    %rsi,%rdi
    6fe3:	e8 a8 d6 ff ff       	callq  4690 <jerasure_free_schedule>
  free(ptrs);
    6fe8:	4c 89 ef             	mov    %r13,%rdi
    6feb:	e8 90 a2 ff ff       	callq  1280 <free@plt>
  return 0;
    6ff0:	31 c0                	xor    %eax,%eax
}
    6ff2:	48 83 c4 18          	add    $0x18,%rsp
    6ff6:	5b                   	pop    %rbx
    6ff7:	5d                   	pop    %rbp
    6ff8:	41 5c                	pop    %r12
    6ffa:	41 5d                	pop    %r13
    6ffc:	41 5e                	pop    %r14
    6ffe:	41 5f                	pop    %r15
    7000:	c3                   	retq   
  if (ptrs == NULL) return -1;
    7001:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    7006:	eb ea                	jmp    6ff2 <jerasure_schedule_decode_lazy+0xd2>
    free(ptrs);
    7008:	4c 89 ef             	mov    %r13,%rdi
    700b:	e8 70 a2 ff ff       	callq  1280 <free@plt>
    return -1;
    7010:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    7015:	eb db                	jmp    6ff2 <jerasure_schedule_decode_lazy+0xd2>
    7017:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    701e:	00 00 

0000000000007020 <jerasure_generate_schedule_cache>:
{
    7020:	f3 0f 1e fa          	endbr64 
    7024:	41 57                	push   %r15
    7026:	41 56                	push   %r14
    7028:	41 55                	push   %r13
    702a:	41 54                	push   %r12
    702c:	55                   	push   %rbp
    702d:	53                   	push   %rbx
    702e:	48 83 ec 78          	sub    $0x78,%rsp
    7032:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    7039:	00 00 
    703b:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
    7040:	31 c0                	xor    %eax,%eax
  if (m != 2) return NULL;
    7042:	83 fe 02             	cmp    $0x2,%esi
    7045:	0f 85 65 01 00 00    	jne    71b0 <jerasure_generate_schedule_cache+0x190>
  scache = talloc(int **, (k+m)*(k+m+1));
    704b:	8d 5f 02             	lea    0x2(%rdi),%ebx
    704e:	89 fd                	mov    %edi,%ebp
    7050:	8d 7f 03             	lea    0x3(%rdi),%edi
    7053:	0f af fb             	imul   %ebx,%edi
    7056:	41 89 d6             	mov    %edx,%r14d
    7059:	49 89 cf             	mov    %rcx,%r15
    705c:	48 63 ff             	movslq %edi,%rdi
    705f:	48 c1 e7 03          	shl    $0x3,%rdi
    7063:	45 89 c5             	mov    %r8d,%r13d
    7066:	e8 95 a3 ff ff       	callq  1400 <malloc@plt>
    706b:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  if (scache == NULL) return NULL;
    7070:	48 85 c0             	test   %rax,%rax
    7073:	0f 84 37 01 00 00    	je     71b0 <jerasure_generate_schedule_cache+0x190>
  for (e1 = 0; e1 < k+m; e1++) {
    7079:	85 db                	test   %ebx,%ebx
    707b:	0f 8e 38 01 00 00    	jle    71b9 <jerasure_generate_schedule_cache+0x199>
    erasures[0] = e1;
    7081:	48 63 db             	movslq %ebx,%rbx
    7084:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    7089:	48 8d 04 dd 08 00 00 	lea    0x8(,%rbx,8),%rax
    7090:	00 
    7091:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    7096:	48 8d 04 dd 00 00 00 	lea    0x0(,%rbx,8),%rax
    709d:	00 
    709e:	48 89 c6             	mov    %rax,%rsi
    70a1:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    70a6:	48 89 c8             	mov    %rcx,%rax
    70a9:	48 01 f0             	add    %rsi,%rax
    70ac:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    70b1:	8d 45 01             	lea    0x1(%rbp),%eax
    70b4:	48 ff c0             	inc    %rax
    70b7:	c7 44 24 5c 00 00 00 	movl   $0x0,0x5c(%rsp)
    70be:	00 
    for (e2 = 0; e2 < e1; e2++) {
    70bf:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    erasures[0] = e1;
    70c4:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
    70c9:	48 c7 44 24 10 01 00 	movq   $0x1,0x10(%rsp)
    70d0:	00 00 
    70d2:	89 6c 24 34          	mov    %ebp,0x34(%rsp)
    70d6:	4c 8d 44 24 5c       	lea    0x5c(%rsp),%r8
    70db:	4d 89 c4             	mov    %r8,%r12
    70de:	66 90                	xchg   %ax,%ax
    scache[e1*(k+m)+e1] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    70e0:	8b 7c 24 34          	mov    0x34(%rsp),%edi
    70e4:	4c 89 f9             	mov    %r15,%rcx
    70e7:	be 02 00 00 00       	mov    $0x2,%esi
    70ec:	45 89 e9             	mov    %r13d,%r9d
    70ef:	4d 89 e0             	mov    %r12,%r8
    70f2:	44 89 f2             	mov    %r14d,%edx
    erasures[1] = -1;
    70f5:	c7 44 24 60 ff ff ff 	movl   $0xffffffff,0x60(%rsp)
    70fc:	ff 
    scache[e1*(k+m)+e1] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    70fd:	e8 3e f7 ff ff       	callq  6840 <jerasure_generate_decoding_schedule>
    7102:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  for (e1 = 0; e1 < k+m; e1++) {
    7107:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
    scache[e1*(k+m)+e1] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    710c:	48 89 01             	mov    %rax,(%rcx)
  for (e1 = 0; e1 < k+m; e1++) {
    710f:	48 39 74 24 10       	cmp    %rsi,0x10(%rsp)
    7114:	0f 84 9f 00 00 00    	je     71b9 <jerasure_generate_schedule_cache+0x199>
    erasures[0] = e1;
    711a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    711f:	89 44 24 5c          	mov    %eax,0x5c(%rsp)
    for (e2 = 0; e2 < e1; e2++) {
    7123:	85 c0                	test   %eax,%eax
    7125:	7e 64                	jle    718b <jerasure_generate_schedule_cache+0x16b>
    7127:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    712c:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    7131:	4d 89 e0             	mov    %r12,%r8
    7134:	44 8b 64 24 34       	mov    0x34(%rsp),%r12d
    7139:	48 8d 2c f8          	lea    (%rax,%rdi,8),%rbp
    713d:	31 db                	xor    %ebx,%ebx
    713f:	90                   	nop
      scache[e1*(k+m)+e2] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    7140:	44 89 f2             	mov    %r14d,%edx
    7143:	45 89 e9             	mov    %r13d,%r9d
    7146:	4c 89 f9             	mov    %r15,%rcx
    7149:	be 02 00 00 00       	mov    $0x2,%esi
    714e:	44 89 e7             	mov    %r12d,%edi
      erasures[1] = e2;
    7151:	89 5c 24 60          	mov    %ebx,0x60(%rsp)
      scache[e1*(k+m)+e2] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    7155:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
      erasures[2] = -1;
    715a:	c7 44 24 64 ff ff ff 	movl   $0xffffffff,0x64(%rsp)
    7161:	ff 
      scache[e1*(k+m)+e2] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    7162:	e8 d9 f6 ff ff       	callq  6840 <jerasure_generate_decoding_schedule>
    7167:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
    for (e2 = 0; e2 < e1; e2++) {
    716c:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
      scache[e1*(k+m)+e2] = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    7171:	48 89 04 da          	mov    %rax,(%rdx,%rbx,8)
      scache[e2*(k+m)+e1] = scache[e1*(k+m)+e2];
    7175:	48 ff c3             	inc    %rbx
    7178:	48 89 45 00          	mov    %rax,0x0(%rbp)
    for (e2 = 0; e2 < e1; e2++) {
    717c:	48 03 6c 24 20       	add    0x20(%rsp),%rbp
    7181:	48 3b 5c 24 10       	cmp    0x10(%rsp),%rbx
    7186:	75 b8                	jne    7140 <jerasure_generate_schedule_cache+0x120>
    7188:	4d 89 c4             	mov    %r8,%r12
    718b:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    7190:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
    7195:	48 ff 44 24 10       	incq   0x10(%rsp)
    719a:	48 01 4c 24 28       	add    %rcx,0x28(%rsp)
    719f:	48 01 74 24 18       	add    %rsi,0x18(%rsp)
    71a4:	e9 37 ff ff ff       	jmpq   70e0 <jerasure_generate_schedule_cache+0xc0>
    71a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  if (m != 2) return NULL;
    71b0:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
    71b7:	00 00 
}
    71b9:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
    71be:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    71c5:	00 00 
    71c7:	75 14                	jne    71dd <jerasure_generate_schedule_cache+0x1bd>
    71c9:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    71ce:	48 83 c4 78          	add    $0x78,%rsp
    71d2:	5b                   	pop    %rbx
    71d3:	5d                   	pop    %rbp
    71d4:	41 5c                	pop    %r12
    71d6:	41 5d                	pop    %r13
    71d8:	41 5e                	pop    %r14
    71da:	41 5f                	pop    %r15
    71dc:	c3                   	retq   
    71dd:	e8 2e a1 ff ff       	callq  1310 <__stack_chk_fail@plt>
    71e2:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    71e9:	00 00 00 00 
    71ed:	0f 1f 00             	nopl   (%rax)

00000000000071f0 <jerasure_bitmatrix_encode>:

void jerasure_bitmatrix_encode(int k, int m, int w, int *bitmatrix,
                            char **data_ptrs, char **coding_ptrs, int size, int packetsize)
{
    71f0:	f3 0f 1e fa          	endbr64 
    71f4:	41 57                	push   %r15
    71f6:	41 56                	push   %r14
    71f8:	41 55                	push   %r13
    71fa:	41 54                	push   %r12
    71fc:	55                   	push   %rbp
    71fd:	53                   	push   %rbx
    71fe:	48 83 ec 18          	sub    $0x18,%rsp
  int i, j, x, y, sptr, pstarted, index;
  char *dptr, *pptr;

  if (packetsize%sizeof(long) != 0) {
    7202:	f6 44 24 58 07       	testb  $0x7,0x58(%rsp)
    7207:	0f 85 9a 00 00 00    	jne    72a7 <jerasure_bitmatrix_encode+0xb7>
    720d:	4d 89 c6             	mov    %r8,%r14
    fprintf(stderr, "jerasure_bitmatrix_encode - packetsize(%d) %c sizeof(long) != 0\n", packetsize, '%');
    exit(1);
  }
  if (size%(packetsize*w) != 0) {
    7210:	44 8b 44 24 58       	mov    0x58(%rsp),%r8d
    7215:	8b 44 24 50          	mov    0x50(%rsp),%eax
    7219:	44 0f af c2          	imul   %edx,%r8d
    721d:	41 89 d4             	mov    %edx,%r12d
    7220:	99                   	cltd   
    7221:	41 f7 f8             	idiv   %r8d
    7224:	85 d2                	test   %edx,%edx
    7226:	0f 85 a9 00 00 00    	jne    72d5 <jerasure_bitmatrix_encode+0xe5>
    fprintf(stderr, "jerasure_bitmatrix_encode - size(%d) %c (packetsize(%d)*w(%d))) != 0\n", 
         size, '%', packetsize, w);
    exit(1);
  }

  for (i = 0; i < m; i++) {
    722c:	85 f6                	test   %esi,%esi
    722e:	7e 68                	jle    7298 <jerasure_bitmatrix_encode+0xa8>
    7230:	44 89 e3             	mov    %r12d,%ebx
    7233:	0f af df             	imul   %edi,%ebx
    7236:	89 fd                	mov    %edi,%ebp
    7238:	4d 89 cf             	mov    %r9,%r15
    723b:	41 0f af dc          	imul   %r12d,%ebx
    723f:	41 89 fd             	mov    %edi,%r13d
    7242:	48 63 db             	movslq %ebx,%rbx
    7245:	48 8d 04 9d 00 00 00 	lea    0x0(,%rbx,4),%rax
    724c:	00 
    724d:	48 89 04 24          	mov    %rax,(%rsp)
    7251:	8d 04 3e             	lea    (%rsi,%rdi,1),%eax
    7254:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    7258:	48 89 cb             	mov    %rcx,%rbx
    725b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    jerasure_bitmatrix_dotprod(k, w, bitmatrix+i*k*w*w, NULL, k+i, data_ptrs, coding_ptrs, size, packetsize);
    7260:	48 83 ec 08          	sub    $0x8,%rsp
    7264:	8b 44 24 60          	mov    0x60(%rsp),%eax
    7268:	45 89 e8             	mov    %r13d,%r8d
    726b:	50                   	push   %rax
    726c:	48 89 da             	mov    %rbx,%rdx
    726f:	4d 89 f1             	mov    %r14,%r9
    7272:	8b 44 24 60          	mov    0x60(%rsp),%eax
    7276:	31 c9                	xor    %ecx,%ecx
    7278:	50                   	push   %rax
    7279:	44 89 e6             	mov    %r12d,%esi
    727c:	89 ef                	mov    %ebp,%edi
    727e:	41 57                	push   %r15
    7280:	41 ff c5             	inc    %r13d
    7283:	e8 38 c5 ff ff       	callq  37c0 <jerasure_bitmatrix_dotprod>
  for (i = 0; i < m; i++) {
    7288:	48 03 5c 24 20       	add    0x20(%rsp),%rbx
    728d:	48 83 c4 20          	add    $0x20,%rsp
    7291:	44 3b 6c 24 0c       	cmp    0xc(%rsp),%r13d
    7296:	75 c8                	jne    7260 <jerasure_bitmatrix_encode+0x70>
  }
}
    7298:	48 83 c4 18          	add    $0x18,%rsp
    729c:	5b                   	pop    %rbx
    729d:	5d                   	pop    %rbp
    729e:	41 5c                	pop    %r12
    72a0:	41 5d                	pop    %r13
    72a2:	41 5e                	pop    %r14
    72a4:	41 5f                	pop    %r15
    72a6:	c3                   	retq   
    72a7:	48 8b 3d 92 9e 00 00 	mov    0x9e92(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    72ae:	8b 4c 24 58          	mov    0x58(%rsp),%ecx
    72b2:	41 b8 25 00 00 00    	mov    $0x25,%r8d
    72b8:	48 8d 15 31 35 00 00 	lea    0x3531(%rip),%rdx        # a7f0 <__PRETTY_FUNCTION__.5741+0x117>
    72bf:	be 01 00 00 00       	mov    $0x1,%esi
    72c4:	31 c0                	xor    %eax,%eax
    72c6:	e8 b5 a1 ff ff       	callq  1480 <__fprintf_chk@plt>
    exit(1);
    72cb:	bf 01 00 00 00       	mov    $0x1,%edi
    72d0:	e8 8b a1 ff ff       	callq  1460 <exit@plt>
    72d5:	50                   	push   %rax
    72d6:	48 8b 3d 63 9e 00 00 	mov    0x9e63(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    72dd:	41 b8 25 00 00 00    	mov    $0x25,%r8d
    72e3:	41 54                	push   %r12
    72e5:	48 8d 15 4c 35 00 00 	lea    0x354c(%rip),%rdx        # a838 <__PRETTY_FUNCTION__.5741+0x15f>
    72ec:	be 01 00 00 00       	mov    $0x1,%esi
    72f1:	44 8b 4c 24 68       	mov    0x68(%rsp),%r9d
    72f6:	8b 4c 24 60          	mov    0x60(%rsp),%ecx
    72fa:	31 c0                	xor    %eax,%eax
    72fc:	e8 7f a1 ff ff       	callq  1480 <__fprintf_chk@plt>
    exit(1);
    7301:	bf 01 00 00 00       	mov    $0x1,%edi
    7306:	e8 55 a1 ff ff       	callq  1460 <exit@plt>
    730b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000007310 <galois_create_log_tables.part.0>:
    7310:	41 56                	push   %r14
    7312:	48 8d 05 87 38 00 00 	lea    0x3887(%rip),%rax        # aba0 <nw>
    7319:	4c 8d 35 00 a2 00 00 	lea    0xa200(%rip),%r14        # 11520 <galois_log_tables>
    7320:	41 55                	push   %r13
    7322:	41 54                	push   %r12
    7324:	4c 63 e7             	movslq %edi,%r12
    7327:	4e 63 2c a0          	movslq (%rax,%r12,4),%r13
    732b:	55                   	push   %rbp
    732c:	4a 8d 3c ad 00 00 00 	lea    0x0(,%r13,4),%rdi
    7333:	00 
    7334:	53                   	push   %rbx
    7335:	e8 c6 a0 ff ff       	callq  1400 <malloc@plt>
    733a:	4b 89 04 e6          	mov    %rax,(%r14,%r12,8)
    733e:	48 85 c0             	test   %rax,%rax
    7341:	0f 84 31 01 00 00    	je     7478 <galois_create_log_tables.part.0+0x168>
    7347:	4b 8d 7c 6d 00       	lea    0x0(%r13,%r13,2),%rdi
    734c:	48 c1 e7 02          	shl    $0x2,%rdi
    7350:	4c 89 eb             	mov    %r13,%rbx
    7353:	48 89 c5             	mov    %rax,%rbp
    7356:	4c 8d 2d a3 a0 00 00 	lea    0xa0a3(%rip),%r13        # 11400 <galois_ilog_tables>
    735d:	e8 9e a0 ff ff       	callq  1400 <malloc@plt>
    7362:	4b 89 44 e5 00       	mov    %rax,0x0(%r13,%r12,8)
    7367:	48 85 c0             	test   %rax,%rax
    736a:	0f 84 f3 00 00 00    	je     7463 <galois_create_log_tables.part.0+0x153>
    7370:	48 8d 15 89 37 00 00 	lea    0x3789(%rip),%rdx        # ab00 <nwm1>
    7377:	42 8b 34 a2          	mov    (%rdx,%r12,4),%esi
    737b:	8d 7b ff             	lea    -0x1(%rbx),%edi
    737e:	31 d2                	xor    %edx,%edx
    7380:	85 db                	test   %ebx,%ebx
    7382:	7e 1a                	jle    739e <galois_create_log_tables.part.0+0x8e>
    7384:	0f 1f 40 00          	nopl   0x0(%rax)
    7388:	48 89 d1             	mov    %rdx,%rcx
    738b:	89 74 95 00          	mov    %esi,0x0(%rbp,%rdx,4)
    738f:	c7 04 90 00 00 00 00 	movl   $0x0,(%rax,%rdx,4)
    7396:	48 ff c2             	inc    %rdx
    7399:	48 39 cf             	cmp    %rcx,%rdi
    739c:	75 ea                	jne    7388 <galois_create_log_tables.part.0+0x78>
    739e:	48 63 ce             	movslq %esi,%rcx
    73a1:	85 f6                	test   %esi,%esi
    73a3:	7e 76                	jle    741b <galois_create_log_tables.part.0+0x10b>
    73a5:	48 89 c2             	mov    %rax,%rdx
    73a8:	48 89 c7             	mov    %rax,%rdi
    73ab:	31 c9                	xor    %ecx,%ecx
    73ad:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    73b3:	4c 8d 35 26 39 00 00 	lea    0x3926(%rip),%r14        # ace0 <prim_poly>
    73ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    73c0:	4d 63 c8             	movslq %r8d,%r9
    73c3:	4e 8d 54 8d 00       	lea    0x0(%rbp,%r9,4),%r10
    73c8:	45 8b 0a             	mov    (%r10),%r9d
    73cb:	47 8d 1c 00          	lea    (%r8,%r8,1),%r11d
    73cf:	41 39 f1             	cmp    %esi,%r9d
    73d2:	75 5b                	jne    742f <galois_create_log_tables.part.0+0x11f>
    73d4:	44 89 07             	mov    %r8d,(%rdi)
    73d7:	41 89 0a             	mov    %ecx,(%r10)
    73da:	45 89 d8             	mov    %r11d,%r8d
    73dd:	44 85 db             	test   %r11d,%ebx
    73e0:	74 07                	je     73e9 <galois_create_log_tables.part.0+0xd9>
    73e2:	47 33 04 a6          	xor    (%r14,%r12,4),%r8d
    73e6:	41 21 f0             	and    %esi,%r8d
    73e9:	ff c1                	inc    %ecx
    73eb:	48 83 c7 04          	add    $0x4,%rdi
    73ef:	39 f1                	cmp    %esi,%ecx
    73f1:	75 cd                	jne    73c0 <galois_create_log_tables.part.0+0xb0>
    73f3:	8d 3c 09             	lea    (%rcx,%rcx,1),%edi
    73f6:	8d 71 ff             	lea    -0x1(%rcx),%esi
    73f9:	4c 8d 44 b0 04       	lea    0x4(%rax,%rsi,4),%r8
    73fe:	48 63 c9             	movslq %ecx,%rcx
    7401:	48 63 ff             	movslq %edi,%rdi
    7404:	0f 1f 40 00          	nopl   0x0(%rax)
    7408:	8b 32                	mov    (%rdx),%esi
    740a:	89 34 8a             	mov    %esi,(%rdx,%rcx,4)
    740d:	8b 32                	mov    (%rdx),%esi
    740f:	89 34 ba             	mov    %esi,(%rdx,%rdi,4)
    7412:	48 83 c2 04          	add    $0x4,%rdx
    7416:	49 39 d0             	cmp    %rdx,%r8
    7419:	75 ed                	jne    7408 <galois_create_log_tables.part.0+0xf8>
    741b:	48 8d 04 88          	lea    (%rax,%rcx,4),%rax
    741f:	4b 89 44 e5 00       	mov    %rax,0x0(%r13,%r12,8)
    7424:	31 c0                	xor    %eax,%eax
    7426:	5b                   	pop    %rbx
    7427:	5d                   	pop    %rbp
    7428:	41 5c                	pop    %r12
    742a:	41 5d                	pop    %r13
    742c:	41 5e                	pop    %r14
    742e:	c3                   	retq   
    742f:	48 8d 05 aa 38 00 00 	lea    0x38aa(%rip),%rax        # ace0 <prim_poly>
    7436:	46 33 1c a0          	xor    (%rax,%r12,4),%r11d
    743a:	41 53                	push   %r11
    743c:	48 8d 15 3d 34 00 00 	lea    0x343d(%rip),%rdx        # a880 <__PRETTY_FUNCTION__.5741+0x1a7>
    7443:	be 01 00 00 00       	mov    $0x1,%esi
    7448:	8b 07                	mov    (%rdi),%eax
    744a:	48 8b 3d ef 9c 00 00 	mov    0x9cef(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    7451:	50                   	push   %rax
    7452:	31 c0                	xor    %eax,%eax
    7454:	e8 27 a0 ff ff       	callq  1480 <__fprintf_chk@plt>
    7459:	bf 01 00 00 00       	mov    $0x1,%edi
    745e:	e8 fd 9f ff ff       	callq  1460 <exit@plt>
    7463:	48 89 ef             	mov    %rbp,%rdi
    7466:	e8 15 9e ff ff       	callq  1280 <free@plt>
    746b:	4b c7 04 e6 00 00 00 	movq   $0x0,(%r14,%r12,8)
    7472:	00 
    7473:	83 c8 ff             	or     $0xffffffff,%eax
    7476:	eb ae                	jmp    7426 <galois_create_log_tables.part.0+0x116>
    7478:	83 c8 ff             	or     $0xffffffff,%eax
    747b:	eb a9                	jmp    7426 <galois_create_log_tables.part.0+0x116>
    747d:	0f 1f 00             	nopl   (%rax)

0000000000007480 <galois_create_log_tables>:
    7480:	f3 0f 1e fa          	endbr64 
    7484:	83 ff 1e             	cmp    $0x1e,%edi
    7487:	7f 27                	jg     74b0 <galois_create_log_tables+0x30>
    7489:	48 63 d7             	movslq %edi,%rdx
    748c:	48 8d 05 8d a0 00 00 	lea    0xa08d(%rip),%rax        # 11520 <galois_log_tables>
    7493:	45 31 c0             	xor    %r8d,%r8d
    7496:	48 83 3c d0 00       	cmpq   $0x0,(%rax,%rdx,8)
    749b:	74 0b                	je     74a8 <galois_create_log_tables+0x28>
    749d:	44 89 c0             	mov    %r8d,%eax
    74a0:	c3                   	retq   
    74a1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    74a8:	e9 63 fe ff ff       	jmpq   7310 <galois_create_log_tables.part.0>
    74ad:	0f 1f 00             	nopl   (%rax)
    74b0:	41 b8 ff ff ff ff    	mov    $0xffffffff,%r8d
    74b6:	eb e5                	jmp    749d <galois_create_log_tables+0x1d>
    74b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    74bf:	00 

00000000000074c0 <galois_logtable_multiply>:
    74c0:	f3 0f 1e fa          	endbr64 
    74c4:	85 ff                	test   %edi,%edi
    74c6:	74 38                	je     7500 <galois_logtable_multiply+0x40>
    74c8:	85 f6                	test   %esi,%esi
    74ca:	74 34                	je     7500 <galois_logtable_multiply+0x40>
    74cc:	48 63 d2             	movslq %edx,%rdx
    74cf:	48 8d 05 4a a0 00 00 	lea    0xa04a(%rip),%rax        # 11520 <galois_log_tables>
    74d6:	48 8b 0c d0          	mov    (%rax,%rdx,8),%rcx
    74da:	48 63 f6             	movslq %esi,%rsi
    74dd:	8b 04 b1             	mov    (%rcx,%rsi,4),%eax
    74e0:	48 63 ff             	movslq %edi,%rdi
    74e3:	03 04 b9             	add    (%rcx,%rdi,4),%eax
    74e6:	48 8d 0d 13 9f 00 00 	lea    0x9f13(%rip),%rcx        # 11400 <galois_ilog_tables>
    74ed:	48 8b 14 d1          	mov    (%rcx,%rdx,8),%rdx
    74f1:	48 98                	cltq   
    74f3:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    74f6:	c3                   	retq   
    74f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    74fe:	00 00 
    7500:	31 c0                	xor    %eax,%eax
    7502:	c3                   	retq   
    7503:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    750a:	00 00 00 00 
    750e:	66 90                	xchg   %ax,%ax

0000000000007510 <galois_logtable_divide>:
    7510:	f3 0f 1e fa          	endbr64 
    7514:	85 f6                	test   %esi,%esi
    7516:	74 38                	je     7550 <galois_logtable_divide+0x40>
    7518:	31 c0                	xor    %eax,%eax
    751a:	85 ff                	test   %edi,%edi
    751c:	74 37                	je     7555 <galois_logtable_divide+0x45>
    751e:	48 63 d2             	movslq %edx,%rdx
    7521:	48 8d 05 f8 9f 00 00 	lea    0x9ff8(%rip),%rax        # 11520 <galois_log_tables>
    7528:	48 8b 0c d0          	mov    (%rax,%rdx,8),%rcx
    752c:	48 63 ff             	movslq %edi,%rdi
    752f:	8b 04 b9             	mov    (%rcx,%rdi,4),%eax
    7532:	48 63 f6             	movslq %esi,%rsi
    7535:	2b 04 b1             	sub    (%rcx,%rsi,4),%eax
    7538:	48 8d 0d c1 9e 00 00 	lea    0x9ec1(%rip),%rcx        # 11400 <galois_ilog_tables>
    753f:	48 8b 14 d1          	mov    (%rcx,%rdx,8),%rdx
    7543:	48 98                	cltq   
    7545:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    7548:	c3                   	retq   
    7549:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7550:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    7555:	c3                   	retq   
    7556:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    755d:	00 00 00 

0000000000007560 <galois_create_mult_tables>:
    7560:	f3 0f 1e fa          	endbr64 
    7564:	83 ff 0d             	cmp    $0xd,%edi
    7567:	0f 8f ed 01 00 00    	jg     775a <galois_create_mult_tables+0x1fa>
    756d:	41 57                	push   %r15
    756f:	41 56                	push   %r14
    7571:	4c 63 f7             	movslq %edi,%r14
    7574:	41 55                	push   %r13
    7576:	41 89 fd             	mov    %edi,%r13d
    7579:	41 54                	push   %r12
    757b:	55                   	push   %rbp
    757c:	53                   	push   %rbx
    757d:	48 8d 1d 5c 9d 00 00 	lea    0x9d5c(%rip),%rbx        # 112e0 <galois_mult_tables>
    7584:	48 83 ec 18          	sub    $0x18,%rsp
    7588:	4a 83 3c f3 00       	cmpq   $0x0,(%rbx,%r14,8)
    758d:	74 11                	je     75a0 <galois_create_mult_tables+0x40>
    758f:	31 c0                	xor    %eax,%eax
    7591:	48 83 c4 18          	add    $0x18,%rsp
    7595:	5b                   	pop    %rbx
    7596:	5d                   	pop    %rbp
    7597:	41 5c                	pop    %r12
    7599:	41 5d                	pop    %r13
    759b:	41 5e                	pop    %r14
    759d:	41 5f                	pop    %r15
    759f:	c3                   	retq   
    75a0:	48 8d 05 f9 35 00 00 	lea    0x35f9(%rip),%rax        # aba0 <nw>
    75a7:	4e 63 0c b0          	movslq (%rax,%r14,4),%r9
    75ab:	4d 89 cc             	mov    %r9,%r12
    75ae:	4d 0f af e1          	imul   %r9,%r12
    75b2:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
    75b7:	4d 89 cf             	mov    %r9,%r15
    75ba:	49 c1 e4 02          	shl    $0x2,%r12
    75be:	4c 89 e7             	mov    %r12,%rdi
    75c1:	e8 3a 9e ff ff       	callq  1400 <malloc@plt>
    75c6:	4a 89 04 f3          	mov    %rax,(%rbx,%r14,8)
    75ca:	48 89 c5             	mov    %rax,%rbp
    75cd:	48 85 c0             	test   %rax,%rax
    75d0:	0f 84 62 01 00 00    	je     7738 <galois_create_mult_tables+0x1d8>
    75d6:	4c 89 e7             	mov    %r12,%rdi
    75d9:	e8 22 9e ff ff       	callq  1400 <malloc@plt>
    75de:	48 85 c0             	test   %rax,%rax
    75e1:	48 8d 0d d8 9b 00 00 	lea    0x9bd8(%rip),%rcx        # 111c0 <galois_div_tables>
    75e8:	4a 89 04 f1          	mov    %rax,(%rcx,%r14,8)
    75ec:	4c 8b 4c 24 08       	mov    0x8(%rsp),%r9
    75f1:	49 89 c4             	mov    %rax,%r12
    75f4:	0f 84 48 01 00 00    	je     7742 <galois_create_mult_tables+0x1e2>
    75fa:	48 8d 15 1f 9f 00 00 	lea    0x9f1f(%rip),%rdx        # 11520 <galois_log_tables>
    7601:	4a 83 3c f2 00       	cmpq   $0x0,(%rdx,%r14,8)
    7606:	0f 84 cb 00 00 00    	je     76d7 <galois_create_mult_tables+0x177>
    760c:	c7 45 00 00 00 00 00 	movl   $0x0,0x0(%rbp)
    7613:	41 c7 04 24 ff ff ff 	movl   $0xffffffff,(%r12)
    761a:	ff 
    761b:	41 83 ff 01          	cmp    $0x1,%r15d
    761f:	0f 8e 6a ff ff ff    	jle    758f <galois_create_mult_tables+0x2f>
    7625:	41 8d 4f fe          	lea    -0x2(%r15),%ecx
    7629:	4c 8d 59 02          	lea    0x2(%rcx),%r11
    762d:	b8 01 00 00 00       	mov    $0x1,%eax
    7632:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7638:	c7 44 85 00 00 00 00 	movl   $0x0,0x0(%rbp,%rax,4)
    763f:	00 
    7640:	41 c7 04 84 00 00 00 	movl   $0x0,(%r12,%rax,4)
    7647:	00 
    7648:	48 ff c0             	inc    %rax
    764b:	4c 39 d8             	cmp    %r11,%rax
    764e:	75 e8                	jne    7638 <galois_create_mult_tables+0xd8>
    7650:	4a 8b 3c f2          	mov    (%rdx,%r14,8),%rdi
    7654:	48 8d 05 a5 9d 00 00 	lea    0x9da5(%rip),%rax        # 11400 <galois_ilog_tables>
    765b:	4e 8b 04 f0          	mov    (%rax,%r14,8),%r8
    765f:	48 8d 5f 04          	lea    0x4(%rdi),%rbx
    7663:	4c 8d 74 8f 08       	lea    0x8(%rdi,%rcx,4),%r14
    7668:	45 8d 6f ff          	lea    -0x1(%r15),%r13d
    766c:	0f 1f 40 00          	nopl   0x0(%rax)
    7670:	49 c1 e1 02          	shl    $0x2,%r9
    7674:	4e 8d 54 0d 00       	lea    0x0(%rbp,%r9,1),%r10
    7679:	4d 01 e1             	add    %r12,%r9
    767c:	41 c7 02 00 00 00 00 	movl   $0x0,(%r10)
    7683:	41 c7 01 ff ff ff ff 	movl   $0xffffffff,(%r9)
    768a:	41 8d 47 01          	lea    0x1(%r15),%eax
    768e:	ba 01 00 00 00       	mov    $0x1,%edx
    7693:	8b 33                	mov    (%rbx),%esi
    7695:	0f 1f 00             	nopl   (%rax)
    7698:	8b 0c 97             	mov    (%rdi,%rdx,4),%ecx
    769b:	01 f1                	add    %esi,%ecx
    769d:	48 63 c9             	movslq %ecx,%rcx
    76a0:	41 8b 0c 88          	mov    (%r8,%rcx,4),%ecx
    76a4:	41 89 0c 92          	mov    %ecx,(%r10,%rdx,4)
    76a8:	89 f1                	mov    %esi,%ecx
    76aa:	2b 0c 97             	sub    (%rdi,%rdx,4),%ecx
    76ad:	48 63 c9             	movslq %ecx,%rcx
    76b0:	41 8b 0c 88          	mov    (%r8,%rcx,4),%ecx
    76b4:	41 89 0c 91          	mov    %ecx,(%r9,%rdx,4)
    76b8:	48 ff c2             	inc    %rdx
    76bb:	4c 39 da             	cmp    %r11,%rdx
    76be:	75 d8                	jne    7698 <galois_create_mult_tables+0x138>
    76c0:	48 83 c3 04          	add    $0x4,%rbx
    76c4:	45 8d 7c 05 00       	lea    0x0(%r13,%rax,1),%r15d
    76c9:	49 39 de             	cmp    %rbx,%r14
    76cc:	0f 84 bd fe ff ff    	je     758f <galois_create_mult_tables+0x2f>
    76d2:	4d 63 cf             	movslq %r15d,%r9
    76d5:	eb 99                	jmp    7670 <galois_create_mult_tables+0x110>
    76d7:	44 89 ef             	mov    %r13d,%edi
    76da:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
    76df:	e8 2c fc ff ff       	callq  7310 <galois_create_log_tables.part.0>
    76e4:	85 c0                	test   %eax,%eax
    76e6:	48 8d 0d d3 9a 00 00 	lea    0x9ad3(%rip),%rcx        # 111c0 <galois_div_tables>
    76ed:	78 19                	js     7708 <galois_create_mult_tables+0x1a8>
    76ef:	4a 8b 2c f3          	mov    (%rbx,%r14,8),%rbp
    76f3:	4e 8b 24 f1          	mov    (%rcx,%r14,8),%r12
    76f7:	4c 8b 4c 24 08       	mov    0x8(%rsp),%r9
    76fc:	48 8d 15 1d 9e 00 00 	lea    0x9e1d(%rip),%rdx        # 11520 <galois_log_tables>
    7703:	e9 04 ff ff ff       	jmpq   760c <galois_create_mult_tables+0xac>
    7708:	4a 8b 3c f3          	mov    (%rbx,%r14,8),%rdi
    770c:	e8 6f 9b ff ff       	callq  1280 <free@plt>
    7711:	48 8d 0d a8 9a 00 00 	lea    0x9aa8(%rip),%rcx        # 111c0 <galois_div_tables>
    7718:	4a 8b 3c f1          	mov    (%rcx,%r14,8),%rdi
    771c:	e8 5f 9b ff ff       	callq  1280 <free@plt>
    7721:	48 8d 0d 98 9a 00 00 	lea    0x9a98(%rip),%rcx        # 111c0 <galois_div_tables>
    7728:	4a c7 04 f3 00 00 00 	movq   $0x0,(%rbx,%r14,8)
    772f:	00 
    7730:	4a c7 04 f1 00 00 00 	movq   $0x0,(%rcx,%r14,8)
    7737:	00 
    7738:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    773d:	e9 4f fe ff ff       	jmpq   7591 <galois_create_mult_tables+0x31>
    7742:	48 89 ef             	mov    %rbp,%rdi
    7745:	e8 36 9b ff ff       	callq  1280 <free@plt>
    774a:	4a c7 04 f3 00 00 00 	movq   $0x0,(%rbx,%r14,8)
    7751:	00 
    7752:	83 c8 ff             	or     $0xffffffff,%eax
    7755:	e9 37 fe ff ff       	jmpq   7591 <galois_create_mult_tables+0x31>
    775a:	83 c8 ff             	or     $0xffffffff,%eax
    775d:	c3                   	retq   
    775e:	66 90                	xchg   %ax,%ax

0000000000007760 <galois_ilog>:
    7760:	f3 0f 1e fa          	endbr64 
    7764:	41 54                	push   %r12
    7766:	4c 8d 25 93 9c 00 00 	lea    0x9c93(%rip),%r12        # 11400 <galois_ilog_tables>
    776d:	55                   	push   %rbp
    776e:	48 63 ee             	movslq %esi,%rbp
    7771:	49 8b 04 ec          	mov    (%r12,%rbp,8),%rax
    7775:	53                   	push   %rbx
    7776:	89 fb                	mov    %edi,%ebx
    7778:	48 85 c0             	test   %rax,%rax
    777b:	74 13                	je     7790 <galois_ilog+0x30>
    777d:	48 63 fb             	movslq %ebx,%rdi
    7780:	5b                   	pop    %rbx
    7781:	5d                   	pop    %rbp
    7782:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    7785:	41 5c                	pop    %r12
    7787:	c3                   	retq   
    7788:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    778f:	00 
    7790:	83 fe 1e             	cmp    $0x1e,%esi
    7793:	7f 1f                	jg     77b4 <galois_ilog+0x54>
    7795:	48 8d 15 84 9d 00 00 	lea    0x9d84(%rip),%rdx        # 11520 <galois_log_tables>
    779c:	48 83 3c ea 00       	cmpq   $0x0,(%rdx,%rbp,8)
    77a1:	75 da                	jne    777d <galois_ilog+0x1d>
    77a3:	89 f7                	mov    %esi,%edi
    77a5:	e8 66 fb ff ff       	callq  7310 <galois_create_log_tables.part.0>
    77aa:	85 c0                	test   %eax,%eax
    77ac:	78 06                	js     77b4 <galois_ilog+0x54>
    77ae:	49 8b 04 ec          	mov    (%r12,%rbp,8),%rax
    77b2:	eb c9                	jmp    777d <galois_ilog+0x1d>
    77b4:	48 8b 0d 85 99 00 00 	mov    0x9985(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    77bb:	48 8d 3d 0e 31 00 00 	lea    0x310e(%rip),%rdi        # a8d0 <__PRETTY_FUNCTION__.5741+0x1f7>
    77c2:	ba 2a 00 00 00       	mov    $0x2a,%edx
    77c7:	be 01 00 00 00       	mov    $0x1,%esi
    77cc:	e8 9f 9c ff ff       	callq  1470 <fwrite@plt>
    77d1:	bf 01 00 00 00       	mov    $0x1,%edi
    77d6:	e8 85 9c ff ff       	callq  1460 <exit@plt>
    77db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000077e0 <galois_log>:
    77e0:	f3 0f 1e fa          	endbr64 
    77e4:	41 54                	push   %r12
    77e6:	4c 63 e6             	movslq %esi,%r12
    77e9:	55                   	push   %rbp
    77ea:	48 8d 2d 2f 9d 00 00 	lea    0x9d2f(%rip),%rbp        # 11520 <galois_log_tables>
    77f1:	4a 8b 44 e5 00       	mov    0x0(%rbp,%r12,8),%rax
    77f6:	53                   	push   %rbx
    77f7:	89 fb                	mov    %edi,%ebx
    77f9:	48 85 c0             	test   %rax,%rax
    77fc:	74 12                	je     7810 <galois_log+0x30>
    77fe:	48 63 fb             	movslq %ebx,%rdi
    7801:	5b                   	pop    %rbx
    7802:	5d                   	pop    %rbp
    7803:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    7806:	41 5c                	pop    %r12
    7808:	c3                   	retq   
    7809:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7810:	83 fe 1e             	cmp    $0x1e,%esi
    7813:	7f 1b                	jg     7830 <galois_log+0x50>
    7815:	89 f7                	mov    %esi,%edi
    7817:	e8 f4 fa ff ff       	callq  7310 <galois_create_log_tables.part.0>
    781c:	85 c0                	test   %eax,%eax
    781e:	78 10                	js     7830 <galois_log+0x50>
    7820:	4a 8b 44 e5 00       	mov    0x0(%rbp,%r12,8),%rax
    7825:	48 63 fb             	movslq %ebx,%rdi
    7828:	5b                   	pop    %rbx
    7829:	5d                   	pop    %rbp
    782a:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    782d:	41 5c                	pop    %r12
    782f:	c3                   	retq   
    7830:	48 8b 0d 09 99 00 00 	mov    0x9909(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    7837:	48 8d 3d c2 30 00 00 	lea    0x30c2(%rip),%rdi        # a900 <__PRETTY_FUNCTION__.5741+0x227>
    783e:	ba 29 00 00 00       	mov    $0x29,%edx
    7843:	be 01 00 00 00       	mov    $0x1,%esi
    7848:	e8 23 9c ff ff       	callq  1470 <fwrite@plt>
    784d:	bf 01 00 00 00       	mov    $0x1,%edi
    7852:	e8 09 9c ff ff       	callq  1460 <exit@plt>
    7857:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    785e:	00 00 

0000000000007860 <galois_shift_multiply>:
    7860:	f3 0f 1e fa          	endbr64 
    7864:	41 54                	push   %r12
    7866:	55                   	push   %rbp
    7867:	53                   	push   %rbx
    7868:	48 81 ec 90 00 00 00 	sub    $0x90,%rsp
    786f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    7876:	00 00 
    7878:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
    787f:	00 
    7880:	31 c0                	xor    %eax,%eax
    7882:	85 d2                	test   %edx,%edx
    7884:	0f 8e c1 00 00 00    	jle    794b <galois_shift_multiply+0xeb>
    788a:	44 8d 5a ff          	lea    -0x1(%rdx),%r11d
    788e:	48 89 e3             	mov    %rsp,%rbx
    7891:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    7897:	48 89 d8             	mov    %rbx,%rax
    789a:	4e 8d 4c 9c 04       	lea    0x4(%rsp,%r11,4),%r9
    789f:	4c 8d 25 3a 34 00 00 	lea    0x343a(%rip),%r12        # ace0 <prim_poly>
    78a6:	4c 63 d2             	movslq %edx,%r10
    78a9:	48 8d 2d 50 32 00 00 	lea    0x3250(%rip),%rbp        # ab00 <nwm1>
    78b0:	c4 42 21 f7 c0       	shlx   %r11d,%r8d,%r8d
    78b5:	0f 1f 00             	nopl   (%rax)
    78b8:	44 89 c1             	mov    %r8d,%ecx
    78bb:	21 f1                	and    %esi,%ecx
    78bd:	89 30                	mov    %esi,(%rax)
    78bf:	01 f6                	add    %esi,%esi
    78c1:	85 c9                	test   %ecx,%ecx
    78c3:	74 09                	je     78ce <galois_shift_multiply+0x6e>
    78c5:	43 33 34 94          	xor    (%r12,%r10,4),%esi
    78c9:	42 23 74 95 00       	and    0x0(%rbp,%r10,4),%esi
    78ce:	48 83 c0 04          	add    $0x4,%rax
    78d2:	4c 39 c8             	cmp    %r9,%rax
    78d5:	75 e1                	jne    78b8 <galois_shift_multiply+0x58>
    78d7:	45 31 c9             	xor    %r9d,%r9d
    78da:	45 31 c0             	xor    %r8d,%r8d
    78dd:	bd 01 00 00 00       	mov    $0x1,%ebp
    78e2:	eb 10                	jmp    78f4 <galois_shift_multiply+0x94>
    78e4:	0f 1f 40 00          	nopl   0x0(%rax)
    78e8:	49 8d 41 01          	lea    0x1(%r9),%rax
    78ec:	4d 39 d9             	cmp    %r11,%r9
    78ef:	74 38                	je     7929 <galois_shift_multiply+0xc9>
    78f1:	49 89 c1             	mov    %rax,%r9
    78f4:	c4 e2 31 f7 c5       	shlx   %r9d,%ebp,%eax
    78f9:	85 f8                	test   %edi,%eax
    78fb:	74 eb                	je     78e8 <galois_shift_multiply+0x88>
    78fd:	46 8b 14 8b          	mov    (%rbx,%r9,4),%r10d
    7901:	31 c9                	xor    %ecx,%ecx
    7903:	b8 01 00 00 00       	mov    $0x1,%eax
    7908:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    790f:	00 
    7910:	44 89 d6             	mov    %r10d,%esi
    7913:	21 c6                	and    %eax,%esi
    7915:	ff c1                	inc    %ecx
    7917:	41 31 f0             	xor    %esi,%r8d
    791a:	01 c0                	add    %eax,%eax
    791c:	39 ca                	cmp    %ecx,%edx
    791e:	75 f0                	jne    7910 <galois_shift_multiply+0xb0>
    7920:	49 8d 41 01          	lea    0x1(%r9),%rax
    7924:	4d 39 d9             	cmp    %r11,%r9
    7927:	75 c8                	jne    78f1 <galois_shift_multiply+0x91>
    7929:	48 8b 84 24 88 00 00 	mov    0x88(%rsp),%rax
    7930:	00 
    7931:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    7938:	00 00 
    793a:	75 14                	jne    7950 <galois_shift_multiply+0xf0>
    793c:	48 81 c4 90 00 00 00 	add    $0x90,%rsp
    7943:	5b                   	pop    %rbx
    7944:	5d                   	pop    %rbp
    7945:	44 89 c0             	mov    %r8d,%eax
    7948:	41 5c                	pop    %r12
    794a:	c3                   	retq   
    794b:	45 31 c0             	xor    %r8d,%r8d
    794e:	eb d9                	jmp    7929 <galois_shift_multiply+0xc9>
    7950:	e8 bb 99 ff ff       	callq  1310 <__stack_chk_fail@plt>
    7955:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    795c:	00 00 00 00 

0000000000007960 <galois_multtable_multiply>:
    7960:	f3 0f 1e fa          	endbr64 
    7964:	48 63 ca             	movslq %edx,%rcx
    7967:	48 8d 05 72 99 00 00 	lea    0x9972(%rip),%rax        # 112e0 <galois_mult_tables>
    796e:	48 8b 04 c8          	mov    (%rax,%rcx,8),%rax
    7972:	c4 e2 69 f7 ff       	shlx   %edx,%edi,%edi
    7977:	09 f7                	or     %esi,%edi
    7979:	48 63 ff             	movslq %edi,%rdi
    797c:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    797f:	c3                   	retq   

0000000000007980 <galois_multtable_divide>:
    7980:	f3 0f 1e fa          	endbr64 
    7984:	48 63 ca             	movslq %edx,%rcx
    7987:	48 8d 05 32 98 00 00 	lea    0x9832(%rip),%rax        # 111c0 <galois_div_tables>
    798e:	48 8b 04 c8          	mov    (%rax,%rcx,8),%rax
    7992:	c4 e2 69 f7 ff       	shlx   %edx,%edi,%edi
    7997:	09 f7                	or     %esi,%edi
    7999:	48 63 ff             	movslq %edi,%rdi
    799c:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    799f:	c3                   	retq   

00000000000079a0 <galois_w08_region_multiply>:
    79a0:	f3 0f 1e fa          	endbr64 
    79a4:	41 54                	push   %r12
    79a6:	55                   	push   %rbp
    79a7:	89 d5                	mov    %edx,%ebp
    79a9:	53                   	push   %rbx
    79aa:	48 89 fb             	mov    %rdi,%rbx
    79ad:	48 83 ec 20          	sub    $0x20,%rsp
    79b1:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    79b8:	00 00 
    79ba:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    79bf:	31 c0                	xor    %eax,%eax
    79c1:	48 8b 05 58 99 00 00 	mov    0x9958(%rip),%rax        # 11320 <galois_mult_tables+0x40>
    79c8:	48 85 c9             	test   %rcx,%rcx
    79cb:	0f 84 83 00 00 00    	je     7a54 <galois_w08_region_multiply+0xb4>
    79d1:	49 89 cc             	mov    %rcx,%r12
    79d4:	48 85 c0             	test   %rax,%rax
    79d7:	0f 84 c0 00 00 00    	je     7a9d <galois_w08_region_multiply+0xfd>
    79dd:	c1 e6 08             	shl    $0x8,%esi
    79e0:	41 89 f1             	mov    %esi,%r9d
    79e3:	45 85 c0             	test   %r8d,%r8d
    79e6:	74 7e                	je     7a66 <galois_w08_region_multiply+0xc6>
    79e8:	85 ed                	test   %ebp,%ebp
    79ea:	7e 4b                	jle    7a37 <galois_w08_region_multiply+0x97>
    79ec:	4c 8b 15 2d 99 00 00 	mov    0x992d(%rip),%r10        # 11320 <galois_mult_tables+0x40>
    79f3:	48 89 df             	mov    %rbx,%rdi
    79f6:	31 d2                	xor    %edx,%edx
    79f8:	4c 8d 44 24 10       	lea    0x10(%rsp),%r8
    79fd:	0f 1f 00             	nopl   (%rax)
    7a00:	31 f6                	xor    %esi,%esi
    7a02:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7a08:	0f b6 04 37          	movzbl (%rdi,%rsi,1),%eax
    7a0c:	44 01 c8             	add    %r9d,%eax
    7a0f:	48 98                	cltq   
    7a11:	41 8b 04 82          	mov    (%r10,%rax,4),%eax
    7a15:	41 88 04 30          	mov    %al,(%r8,%rsi,1)
    7a19:	48 ff c6             	inc    %rsi
    7a1c:	48 83 fe 08          	cmp    $0x8,%rsi
    7a20:	75 e6                	jne    7a08 <galois_w08_region_multiply+0x68>
    7a22:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    7a27:	48 83 c7 08          	add    $0x8,%rdi
    7a2b:	49 31 04 14          	xor    %rax,(%r12,%rdx,1)
    7a2f:	48 83 c2 08          	add    $0x8,%rdx
    7a33:	39 d5                	cmp    %edx,%ebp
    7a35:	7f c9                	jg     7a00 <galois_w08_region_multiply+0x60>
    7a37:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    7a3c:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    7a43:	00 00 
    7a45:	0f 85 c1 00 00 00    	jne    7b0c <galois_w08_region_multiply+0x16c>
    7a4b:	48 83 c4 20          	add    $0x20,%rsp
    7a4f:	5b                   	pop    %rbx
    7a50:	5d                   	pop    %rbp
    7a51:	41 5c                	pop    %r12
    7a53:	c3                   	retq   
    7a54:	48 85 c0             	test   %rax,%rax
    7a57:	0f 84 93 00 00 00    	je     7af0 <galois_w08_region_multiply+0x150>
    7a5d:	c1 e6 08             	shl    $0x8,%esi
    7a60:	41 89 f1             	mov    %esi,%r9d
    7a63:	49 89 dc             	mov    %rbx,%r12
    7a66:	85 ed                	test   %ebp,%ebp
    7a68:	7e cd                	jle    7a37 <galois_w08_region_multiply+0x97>
    7a6a:	48 8b 35 af 98 00 00 	mov    0x98af(%rip),%rsi        # 11320 <galois_mult_tables+0x40>
    7a71:	8d 4d ff             	lea    -0x1(%rbp),%ecx
    7a74:	31 d2                	xor    %edx,%edx
    7a76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    7a7d:	00 00 00 
    7a80:	0f b6 04 13          	movzbl (%rbx,%rdx,1),%eax
    7a84:	44 01 c8             	add    %r9d,%eax
    7a87:	48 98                	cltq   
    7a89:	8b 04 86             	mov    (%rsi,%rax,4),%eax
    7a8c:	41 88 04 14          	mov    %al,(%r12,%rdx,1)
    7a90:	48 89 d0             	mov    %rdx,%rax
    7a93:	48 ff c2             	inc    %rdx
    7a96:	48 39 c8             	cmp    %rcx,%rax
    7a99:	75 e5                	jne    7a80 <galois_w08_region_multiply+0xe0>
    7a9b:	eb 9a                	jmp    7a37 <galois_w08_region_multiply+0x97>
    7a9d:	bf 08 00 00 00       	mov    $0x8,%edi
    7aa2:	44 89 44 24 0c       	mov    %r8d,0xc(%rsp)
    7aa7:	89 74 24 08          	mov    %esi,0x8(%rsp)
    7aab:	e8 b0 fa ff ff       	callq  7560 <galois_create_mult_tables>
    7ab0:	85 c0                	test   %eax,%eax
    7ab2:	8b 74 24 08          	mov    0x8(%rsp),%esi
    7ab6:	44 8b 44 24 0c       	mov    0xc(%rsp),%r8d
    7abb:	0f 89 1c ff ff ff    	jns    79dd <galois_w08_region_multiply+0x3d>
    7ac1:	48 8b 0d 78 96 00 00 	mov    0x9678(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    7ac8:	48 8d 3d 61 2e 00 00 	lea    0x2e61(%rip),%rdi        # a930 <__PRETTY_FUNCTION__.5741+0x257>
    7acf:	ba 41 00 00 00       	mov    $0x41,%edx
    7ad4:	be 01 00 00 00       	mov    $0x1,%esi
    7ad9:	e8 92 99 ff ff       	callq  1470 <fwrite@plt>
    7ade:	bf 01 00 00 00       	mov    $0x1,%edi
    7ae3:	e8 78 99 ff ff       	callq  1460 <exit@plt>
    7ae8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    7aef:	00 
    7af0:	bf 08 00 00 00       	mov    $0x8,%edi
    7af5:	89 74 24 08          	mov    %esi,0x8(%rsp)
    7af9:	e8 62 fa ff ff       	callq  7560 <galois_create_mult_tables>
    7afe:	85 c0                	test   %eax,%eax
    7b00:	8b 74 24 08          	mov    0x8(%rsp),%esi
    7b04:	0f 89 53 ff ff ff    	jns    7a5d <galois_w08_region_multiply+0xbd>
    7b0a:	eb b5                	jmp    7ac1 <galois_w08_region_multiply+0x121>
    7b0c:	e8 ff 97 ff ff       	callq  1310 <__stack_chk_fail@plt>
    7b11:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    7b18:	00 00 00 00 
    7b1c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000007b20 <galois_w16_region_multiply>:
    7b20:	f3 0f 1e fa          	endbr64 
    7b24:	41 55                	push   %r13
    7b26:	41 54                	push   %r12
    7b28:	41 89 d4             	mov    %edx,%r12d
    7b2b:	55                   	push   %rbp
    7b2c:	48 89 cd             	mov    %rcx,%rbp
    7b2f:	53                   	push   %rbx
    7b30:	48 83 ec 38          	sub    $0x38,%rsp
    7b34:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    7b3b:	00 00 
    7b3d:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    7b42:	31 c0                	xor    %eax,%eax
    7b44:	48 85 c9             	test   %rcx,%rcx
    7b47:	48 0f 44 ef          	cmove  %rdi,%rbp
    7b4b:	41 c1 ec 1f          	shr    $0x1f,%r12d
    7b4f:	41 01 d4             	add    %edx,%r12d
    7b52:	41 d1 fc             	sar    %r12d
    7b55:	85 f6                	test   %esi,%esi
    7b57:	0f 84 f3 00 00 00    	je     7c50 <galois_w16_region_multiply+0x130>
    7b5d:	4c 8b 0d 3c 9a 00 00 	mov    0x9a3c(%rip),%r9        # 115a0 <galois_log_tables+0x80>
    7b64:	48 89 fb             	mov    %rdi,%rbx
    7b67:	4d 85 c9             	test   %r9,%r9
    7b6a:	0f 84 16 01 00 00    	je     7c86 <galois_w16_region_multiply+0x166>
    7b70:	4c 63 ee             	movslq %esi,%r13
    7b73:	47 8b 14 a9          	mov    (%r9,%r13,4),%r10d
    7b77:	48 85 c9             	test   %rcx,%rcx
    7b7a:	74 74                	je     7bf0 <galois_w16_region_multiply+0xd0>
    7b7c:	45 85 c0             	test   %r8d,%r8d
    7b7f:	74 6f                	je     7bf0 <galois_w16_region_multiply+0xd0>
    7b81:	83 fa 01             	cmp    $0x1,%edx
    7b84:	7e 4b                	jle    7bd1 <galois_w16_region_multiply+0xb1>
    7b86:	4c 8b 05 f3 98 00 00 	mov    0x98f3(%rip),%r8        # 11480 <galois_ilog_tables+0x80>
    7b8d:	4c 8b 5c 24 20       	mov    0x20(%rsp),%r11
    7b92:	48 89 df             	mov    %rbx,%rdi
    7b95:	31 c9                	xor    %ecx,%ecx
    7b97:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
    7b9c:	0f 1f 40 00          	nopl   0x0(%rax)
    7ba0:	31 c0                	xor    %eax,%eax
    7ba2:	0f b7 14 07          	movzwl (%rdi,%rax,1),%edx
    7ba6:	66 85 d2             	test   %dx,%dx
    7ba9:	0f 85 81 00 00 00    	jne    7c30 <galois_w16_region_multiply+0x110>
    7baf:	31 d2                	xor    %edx,%edx
    7bb1:	66 89 14 06          	mov    %dx,(%rsi,%rax,1)
    7bb5:	48 83 c0 02          	add    $0x2,%rax
    7bb9:	48 83 f8 08          	cmp    $0x8,%rax
    7bbd:	75 e3                	jne    7ba2 <galois_w16_region_multiply+0x82>
    7bbf:	4c 31 5c 4d 00       	xor    %r11,0x0(%rbp,%rcx,2)
    7bc4:	48 83 c1 04          	add    $0x4,%rcx
    7bc8:	48 83 c7 08          	add    $0x8,%rdi
    7bcc:	41 39 cc             	cmp    %ecx,%r12d
    7bcf:	7f cf                	jg     7ba0 <galois_w16_region_multiply+0x80>
    7bd1:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    7bd6:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    7bdd:	00 00 
    7bdf:	0f 85 06 01 00 00    	jne    7ceb <galois_w16_region_multiply+0x1cb>
    7be5:	48 83 c4 38          	add    $0x38,%rsp
    7be9:	5b                   	pop    %rbx
    7bea:	5d                   	pop    %rbp
    7beb:	41 5c                	pop    %r12
    7bed:	41 5d                	pop    %r13
    7bef:	c3                   	retq   
    7bf0:	83 fa 01             	cmp    $0x1,%edx
    7bf3:	7e dc                	jle    7bd1 <galois_w16_region_multiply+0xb1>
    7bf5:	48 8b 0d 84 98 00 00 	mov    0x9884(%rip),%rcx        # 11480 <galois_ilog_tables+0x80>
    7bfc:	31 c0                	xor    %eax,%eax
    7bfe:	eb 0f                	jmp    7c0f <galois_w16_region_multiply+0xef>
    7c00:	31 f6                	xor    %esi,%esi
    7c02:	66 89 74 45 00       	mov    %si,0x0(%rbp,%rax,2)
    7c07:	48 ff c0             	inc    %rax
    7c0a:	41 39 c4             	cmp    %eax,%r12d
    7c0d:	7e c2                	jle    7bd1 <galois_w16_region_multiply+0xb1>
    7c0f:	0f b7 14 43          	movzwl (%rbx,%rax,2),%edx
    7c13:	66 85 d2             	test   %dx,%dx
    7c16:	74 e8                	je     7c00 <galois_w16_region_multiply+0xe0>
    7c18:	41 8b 3c 91          	mov    (%r9,%rdx,4),%edi
    7c1c:	44 01 d7             	add    %r10d,%edi
    7c1f:	48 63 d7             	movslq %edi,%rdx
    7c22:	8b 14 91             	mov    (%rcx,%rdx,4),%edx
    7c25:	66 89 54 45 00       	mov    %dx,0x0(%rbp,%rax,2)
    7c2a:	eb db                	jmp    7c07 <galois_w16_region_multiply+0xe7>
    7c2c:	0f 1f 40 00          	nopl   0x0(%rax)
    7c30:	41 8b 1c 91          	mov    (%r9,%rdx,4),%ebx
    7c34:	44 01 d3             	add    %r10d,%ebx
    7c37:	48 63 d3             	movslq %ebx,%rdx
    7c3a:	41 8b 14 90          	mov    (%r8,%rdx,4),%edx
    7c3e:	66 89 14 06          	mov    %dx,(%rsi,%rax,1)
    7c42:	e9 6e ff ff ff       	jmpq   7bb5 <galois_w16_region_multiply+0x95>
    7c47:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    7c4e:	00 00 
    7c50:	45 85 c0             	test   %r8d,%r8d
    7c53:	0f 85 78 ff ff ff    	jne    7bd1 <galois_w16_region_multiply+0xb1>
    7c59:	4d 63 e4             	movslq %r12d,%r12
    7c5c:	4a 8d 44 65 00       	lea    0x0(%rbp,%r12,2),%rax
    7c61:	48 39 c5             	cmp    %rax,%rbp
    7c64:	0f 83 67 ff ff ff    	jae    7bd1 <galois_w16_region_multiply+0xb1>
    7c6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7c70:	48 c7 45 00 00 00 00 	movq   $0x0,0x0(%rbp)
    7c77:	00 
    7c78:	48 83 c5 08          	add    $0x8,%rbp
    7c7c:	48 39 e8             	cmp    %rbp,%rax
    7c7f:	77 ef                	ja     7c70 <galois_w16_region_multiply+0x150>
    7c81:	e9 4b ff ff ff       	jmpq   7bd1 <galois_w16_region_multiply+0xb1>
    7c86:	bf 10 00 00 00       	mov    $0x10,%edi
    7c8b:	44 89 44 24 1c       	mov    %r8d,0x1c(%rsp)
    7c90:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    7c95:	89 54 24 18          	mov    %edx,0x18(%rsp)
    7c99:	89 74 24 0c          	mov    %esi,0xc(%rsp)
    7c9d:	e8 6e f6 ff ff       	callq  7310 <galois_create_log_tables.part.0>
    7ca2:	85 c0                	test   %eax,%eax
    7ca4:	78 1e                	js     7cc4 <galois_w16_region_multiply+0x1a4>
    7ca6:	4c 8b 0d f3 98 00 00 	mov    0x98f3(%rip),%r9        # 115a0 <galois_log_tables+0x80>
    7cad:	8b 74 24 0c          	mov    0xc(%rsp),%esi
    7cb1:	8b 54 24 18          	mov    0x18(%rsp),%edx
    7cb5:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    7cba:	44 8b 44 24 1c       	mov    0x1c(%rsp),%r8d
    7cbf:	e9 ac fe ff ff       	jmpq   7b70 <galois_w16_region_multiply+0x50>
    7cc4:	48 8b 0d 75 94 00 00 	mov    0x9475(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    7ccb:	48 8d 3d a6 2c 00 00 	lea    0x2ca6(%rip),%rdi        # a978 <__PRETTY_FUNCTION__.5741+0x29f>
    7cd2:	ba 36 00 00 00       	mov    $0x36,%edx
    7cd7:	be 01 00 00 00       	mov    $0x1,%esi
    7cdc:	e8 8f 97 ff ff       	callq  1470 <fwrite@plt>
    7ce1:	bf 01 00 00 00       	mov    $0x1,%edi
    7ce6:	e8 75 97 ff ff       	callq  1460 <exit@plt>
    7ceb:	e8 20 96 ff ff       	callq  1310 <__stack_chk_fail@plt>

0000000000007cf0 <galois_invert_binary_matrix>:
    7cf0:	f3 0f 1e fa          	endbr64 
    7cf4:	85 d2                	test   %edx,%edx
    7cf6:	0f 8e 58 01 00 00    	jle    7e54 <galois_invert_binary_matrix+0x164>
    7cfc:	41 55                	push   %r13
    7cfe:	44 8d 5a ff          	lea    -0x1(%rdx),%r11d
    7d02:	41 89 d1             	mov    %edx,%r9d
    7d05:	41 54                	push   %r12
    7d07:	4d 89 da             	mov    %r11,%r10
    7d0a:	31 c0                	xor    %eax,%eax
    7d0c:	55                   	push   %rbp
    7d0d:	b9 01 00 00 00       	mov    $0x1,%ecx
    7d12:	49 8d 6b 01          	lea    0x1(%r11),%rbp
    7d16:	53                   	push   %rbx
    7d17:	48 83 ec 08          	sub    $0x8,%rsp
    7d1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7d20:	c4 e2 79 f7 d1       	shlx   %eax,%ecx,%edx
    7d25:	89 14 86             	mov    %edx,(%rsi,%rax,4)
    7d28:	48 89 c2             	mov    %rax,%rdx
    7d2b:	48 ff c0             	inc    %rax
    7d2e:	4c 39 da             	cmp    %r11,%rdx
    7d31:	75 ed                	jne    7d20 <galois_invert_binary_matrix+0x30>
    7d33:	49 83 c3 02          	add    $0x2,%r11
    7d37:	b9 01 00 00 00       	mov    $0x1,%ecx
    7d3c:	bb 01 00 00 00       	mov    $0x1,%ebx
    7d41:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7d48:	44 8b 6c 8f fc       	mov    -0x4(%rdi,%rcx,4),%r13d
    7d4d:	44 8d 41 ff          	lea    -0x1(%rcx),%r8d
    7d51:	45 0f a3 c5          	bt     %r8d,%r13d
    7d55:	89 ca                	mov    %ecx,%edx
    7d57:	0f 83 9b 00 00 00    	jae    7df8 <galois_invert_binary_matrix+0x108>
    7d5d:	48 39 e9             	cmp    %rbp,%rcx
    7d60:	74 3e                	je     7da0 <galois_invert_binary_matrix+0xb0>
    7d62:	48 89 c8             	mov    %rcx,%rax
    7d65:	c4 62 39 f7 c3       	shlx   %r8d,%ebx,%r8d
    7d6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7d70:	8b 14 87             	mov    (%rdi,%rax,4),%edx
    7d73:	44 85 c2             	test   %r8d,%edx
    7d76:	74 0e                	je     7d86 <galois_invert_binary_matrix+0x96>
    7d78:	33 54 8f fc          	xor    -0x4(%rdi,%rcx,4),%edx
    7d7c:	89 14 87             	mov    %edx,(%rdi,%rax,4)
    7d7f:	8b 54 8e fc          	mov    -0x4(%rsi,%rcx,4),%edx
    7d83:	31 14 86             	xor    %edx,(%rsi,%rax,4)
    7d86:	48 ff c0             	inc    %rax
    7d89:	41 39 c1             	cmp    %eax,%r9d
    7d8c:	75 e2                	jne    7d70 <galois_invert_binary_matrix+0x80>
    7d8e:	48 ff c1             	inc    %rcx
    7d91:	49 39 cb             	cmp    %rcx,%r11
    7d94:	75 b2                	jne    7d48 <galois_invert_binary_matrix+0x58>
    7d96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    7d9d:	00 00 00 
    7da0:	4d 63 d2             	movslq %r10d,%r10
    7da3:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    7da9:	45 85 d2             	test   %r10d,%r10d
    7dac:	7e 33                	jle    7de1 <galois_invert_binary_matrix+0xf1>
    7dae:	41 8d 4a ff          	lea    -0x1(%r10),%ecx
    7db2:	31 c0                	xor    %eax,%eax
    7db4:	c4 42 29 f7 c1       	shlx   %r10d,%r9d,%r8d
    7db9:	eb 08                	jmp    7dc3 <galois_invert_binary_matrix+0xd3>
    7dbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7dc0:	48 89 d0             	mov    %rdx,%rax
    7dc3:	44 85 04 87          	test   %r8d,(%rdi,%rax,4)
    7dc7:	74 07                	je     7dd0 <galois_invert_binary_matrix+0xe0>
    7dc9:	42 8b 14 96          	mov    (%rsi,%r10,4),%edx
    7dcd:	31 14 86             	xor    %edx,(%rsi,%rax,4)
    7dd0:	48 8d 50 01          	lea    0x1(%rax),%rdx
    7dd4:	48 39 c8             	cmp    %rcx,%rax
    7dd7:	75 e7                	jne    7dc0 <galois_invert_binary_matrix+0xd0>
    7dd9:	49 ff ca             	dec    %r10
    7ddc:	45 85 d2             	test   %r10d,%r10d
    7ddf:	7f cd                	jg     7dae <galois_invert_binary_matrix+0xbe>
    7de1:	44 89 d0             	mov    %r10d,%eax
    7de4:	ff c8                	dec    %eax
    7de6:	79 f1                	jns    7dd9 <galois_invert_binary_matrix+0xe9>
    7de8:	48 83 c4 08          	add    $0x8,%rsp
    7dec:	5b                   	pop    %rbx
    7ded:	5d                   	pop    %rbp
    7dee:	41 5c                	pop    %r12
    7df0:	41 5d                	pop    %r13
    7df2:	c3                   	retq   
    7df3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7df8:	41 39 c9             	cmp    %ecx,%r9d
    7dfb:	7e 26                	jle    7e23 <galois_invert_binary_matrix+0x133>
    7dfd:	48 89 c8             	mov    %rcx,%rax
    7e00:	c4 62 39 f7 e3       	shlx   %r8d,%ebx,%r12d
    7e05:	eb 14                	jmp    7e1b <galois_invert_binary_matrix+0x12b>
    7e07:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    7e0e:	00 00 
    7e10:	8d 50 01             	lea    0x1(%rax),%edx
    7e13:	48 ff c0             	inc    %rax
    7e16:	41 39 c1             	cmp    %eax,%r9d
    7e19:	7e 08                	jle    7e23 <galois_invert_binary_matrix+0x133>
    7e1b:	89 c2                	mov    %eax,%edx
    7e1d:	44 85 24 87          	test   %r12d,(%rdi,%rax,4)
    7e21:	74 ed                	je     7e10 <galois_invert_binary_matrix+0x120>
    7e23:	41 39 d1             	cmp    %edx,%r9d
    7e26:	74 2d                	je     7e55 <galois_invert_binary_matrix+0x165>
    7e28:	48 63 c2             	movslq %edx,%rax
    7e2b:	48 c1 e0 02          	shl    $0x2,%rax
    7e2f:	48 8d 14 07          	lea    (%rdi,%rax,1),%rdx
    7e33:	44 8b 22             	mov    (%rdx),%r12d
    7e36:	48 01 f0             	add    %rsi,%rax
    7e39:	44 89 64 8f fc       	mov    %r12d,-0x4(%rdi,%rcx,4)
    7e3e:	44 89 2a             	mov    %r13d,(%rdx)
    7e41:	8b 54 8e fc          	mov    -0x4(%rsi,%rcx,4),%edx
    7e45:	44 8b 20             	mov    (%rax),%r12d
    7e48:	44 89 64 8e fc       	mov    %r12d,-0x4(%rsi,%rcx,4)
    7e4d:	89 10                	mov    %edx,(%rax)
    7e4f:	e9 09 ff ff ff       	jmpq   7d5d <galois_invert_binary_matrix+0x6d>
    7e54:	c3                   	retq   
    7e55:	48 8b 0d e4 92 00 00 	mov    0x92e4(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    7e5c:	48 8d 3d 4d 2b 00 00 	lea    0x2b4d(%rip),%rdi        # a9b0 <__PRETTY_FUNCTION__.5741+0x2d7>
    7e63:	ba 2e 00 00 00       	mov    $0x2e,%edx
    7e68:	be 01 00 00 00       	mov    $0x1,%esi
    7e6d:	e8 fe 95 ff ff       	callq  1470 <fwrite@plt>
    7e72:	bf 01 00 00 00       	mov    $0x1,%edi
    7e77:	e8 e4 95 ff ff       	callq  1460 <exit@plt>
    7e7c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000007e80 <galois_shift_inverse>:
    7e80:	f3 0f 1e fa          	endbr64 
    7e84:	55                   	push   %rbp
    7e85:	89 f2                	mov    %esi,%edx
    7e87:	48 81 ec 10 01 00 00 	sub    $0x110,%rsp
    7e8e:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    7e95:	00 00 
    7e97:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
    7e9e:	00 
    7e9f:	31 c0                	xor    %eax,%eax
    7ea1:	85 f6                	test   %esi,%esi
    7ea3:	0f 8e 8f 00 00 00    	jle    7f38 <galois_shift_inverse+0xb8>
    7ea9:	8d 4e ff             	lea    -0x1(%rsi),%ecx
    7eac:	48 63 f1             	movslq %ecx,%rsi
    7eaf:	48 8d 05 ea 2c 00 00 	lea    0x2cea(%rip),%rax        # aba0 <nw>
    7eb6:	48 89 e5             	mov    %rsp,%rbp
    7eb9:	89 c9                	mov    %ecx,%ecx
    7ebb:	44 8b 14 b0          	mov    (%rax,%rsi,4),%r10d
    7ebf:	4c 8d 4c 8c 04       	lea    0x4(%rsp,%rcx,4),%r9
    7ec4:	48 89 e8             	mov    %rbp,%rax
    7ec7:	4c 8d 1d 12 2e 00 00 	lea    0x2e12(%rip),%r11        # ace0 <prim_poly>
    7ece:	4c 63 c2             	movslq %edx,%r8
    7ed1:	48 8d 35 28 2c 00 00 	lea    0x2c28(%rip),%rsi        # ab00 <nwm1>
    7ed8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    7edf:	00 
    7ee0:	44 89 d1             	mov    %r10d,%ecx
    7ee3:	21 f9                	and    %edi,%ecx
    7ee5:	89 38                	mov    %edi,(%rax)
    7ee7:	01 ff                	add    %edi,%edi
    7ee9:	85 c9                	test   %ecx,%ecx
    7eeb:	74 08                	je     7ef5 <galois_shift_inverse+0x75>
    7eed:	43 33 3c 83          	xor    (%r11,%r8,4),%edi
    7ef1:	42 23 3c 86          	and    (%rsi,%r8,4),%edi
    7ef5:	48 83 c0 04          	add    $0x4,%rax
    7ef9:	4c 39 c8             	cmp    %r9,%rax
    7efc:	75 e2                	jne    7ee0 <galois_shift_inverse+0x60>
    7efe:	48 8d b4 24 80 00 00 	lea    0x80(%rsp),%rsi
    7f05:	00 
    7f06:	48 89 ef             	mov    %rbp,%rdi
    7f09:	e8 e2 fd ff ff       	callq  7cf0 <galois_invert_binary_matrix>
    7f0e:	48 8b 94 24 08 01 00 	mov    0x108(%rsp),%rdx
    7f15:	00 
    7f16:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
    7f1d:	00 00 
    7f1f:	8b 84 24 80 00 00 00 	mov    0x80(%rsp),%eax
    7f26:	75 15                	jne    7f3d <galois_shift_inverse+0xbd>
    7f28:	48 81 c4 10 01 00 00 	add    $0x110,%rsp
    7f2f:	5d                   	pop    %rbp
    7f30:	c3                   	retq   
    7f31:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7f38:	48 89 e5             	mov    %rsp,%rbp
    7f3b:	eb c1                	jmp    7efe <galois_shift_inverse+0x7e>
    7f3d:	e8 ce 93 ff ff       	callq  1310 <__stack_chk_fail@plt>
    7f42:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    7f49:	00 00 00 00 
    7f4d:	0f 1f 00             	nopl   (%rax)

0000000000007f50 <galois_shift_divide>:
    7f50:	f3 0f 1e fa          	endbr64 
    7f54:	41 54                	push   %r12
    7f56:	48 83 ec 10          	sub    $0x10,%rsp
    7f5a:	85 f6                	test   %esi,%esi
    7f5c:	74 3a                	je     7f98 <galois_shift_divide+0x48>
    7f5e:	41 89 fc             	mov    %edi,%r12d
    7f61:	85 ff                	test   %edi,%edi
    7f63:	75 0b                	jne    7f70 <galois_shift_divide+0x20>
    7f65:	48 83 c4 10          	add    $0x10,%rsp
    7f69:	44 89 e0             	mov    %r12d,%eax
    7f6c:	41 5c                	pop    %r12
    7f6e:	c3                   	retq   
    7f6f:	90                   	nop
    7f70:	89 f7                	mov    %esi,%edi
    7f72:	89 d6                	mov    %edx,%esi
    7f74:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    7f78:	e8 03 ff ff ff       	callq  7e80 <galois_shift_inverse>
    7f7d:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    7f81:	48 83 c4 10          	add    $0x10,%rsp
    7f85:	44 89 e7             	mov    %r12d,%edi
    7f88:	89 c6                	mov    %eax,%esi
    7f8a:	41 5c                	pop    %r12
    7f8c:	e9 cf f8 ff ff       	jmpq   7860 <galois_shift_multiply>
    7f91:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7f98:	41 bc ff ff ff ff    	mov    $0xffffffff,%r12d
    7f9e:	eb c5                	jmp    7f65 <galois_shift_divide+0x15>

0000000000007fa0 <galois_get_mult_table>:
    7fa0:	f3 0f 1e fa          	endbr64 
    7fa4:	41 54                	push   %r12
    7fa6:	55                   	push   %rbp
    7fa7:	48 63 ef             	movslq %edi,%rbp
    7faa:	53                   	push   %rbx
    7fab:	48 8d 1d 2e 93 00 00 	lea    0x932e(%rip),%rbx        # 112e0 <galois_mult_tables>
    7fb2:	4c 8b 24 eb          	mov    (%rbx,%rbp,8),%r12
    7fb6:	4d 85 e4             	test   %r12,%r12
    7fb9:	74 0d                	je     7fc8 <galois_get_mult_table+0x28>
    7fbb:	5b                   	pop    %rbx
    7fbc:	5d                   	pop    %rbp
    7fbd:	4c 89 e0             	mov    %r12,%rax
    7fc0:	41 5c                	pop    %r12
    7fc2:	c3                   	retq   
    7fc3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7fc8:	e8 93 f5 ff ff       	callq  7560 <galois_create_mult_tables>
    7fcd:	85 c0                	test   %eax,%eax
    7fcf:	75 ea                	jne    7fbb <galois_get_mult_table+0x1b>
    7fd1:	4c 8b 24 eb          	mov    (%rbx,%rbp,8),%r12
    7fd5:	5b                   	pop    %rbx
    7fd6:	5d                   	pop    %rbp
    7fd7:	4c 89 e0             	mov    %r12,%rax
    7fda:	41 5c                	pop    %r12
    7fdc:	c3                   	retq   
    7fdd:	0f 1f 00             	nopl   (%rax)

0000000000007fe0 <galois_get_div_table>:
    7fe0:	f3 0f 1e fa          	endbr64 
    7fe4:	41 54                	push   %r12
    7fe6:	48 8d 05 f3 92 00 00 	lea    0x92f3(%rip),%rax        # 112e0 <galois_mult_tables>
    7fed:	53                   	push   %rbx
    7fee:	48 63 df             	movslq %edi,%rbx
    7ff1:	48 83 ec 08          	sub    $0x8,%rsp
    7ff5:	4c 8b 24 d8          	mov    (%rax,%rbx,8),%r12
    7ff9:	4d 85 e4             	test   %r12,%r12
    7ffc:	74 1a                	je     8018 <galois_get_div_table+0x38>
    7ffe:	48 8d 05 bb 91 00 00 	lea    0x91bb(%rip),%rax        # 111c0 <galois_div_tables>
    8005:	4c 8b 24 d8          	mov    (%rax,%rbx,8),%r12
    8009:	48 83 c4 08          	add    $0x8,%rsp
    800d:	5b                   	pop    %rbx
    800e:	4c 89 e0             	mov    %r12,%rax
    8011:	41 5c                	pop    %r12
    8013:	c3                   	retq   
    8014:	0f 1f 40 00          	nopl   0x0(%rax)
    8018:	e8 43 f5 ff ff       	callq  7560 <galois_create_mult_tables>
    801d:	85 c0                	test   %eax,%eax
    801f:	74 dd                	je     7ffe <galois_get_div_table+0x1e>
    8021:	eb e6                	jmp    8009 <galois_get_div_table+0x29>
    8023:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    802a:	00 00 00 00 
    802e:	66 90                	xchg   %ax,%ax

0000000000008030 <galois_get_log_table>:
    8030:	f3 0f 1e fa          	endbr64 
    8034:	41 54                	push   %r12
    8036:	55                   	push   %rbp
    8037:	48 63 ef             	movslq %edi,%rbp
    803a:	53                   	push   %rbx
    803b:	48 8d 1d de 94 00 00 	lea    0x94de(%rip),%rbx        # 11520 <galois_log_tables>
    8042:	4c 8b 24 eb          	mov    (%rbx,%rbp,8),%r12
    8046:	4d 85 e4             	test   %r12,%r12
    8049:	74 0d                	je     8058 <galois_get_log_table+0x28>
    804b:	5b                   	pop    %rbx
    804c:	5d                   	pop    %rbp
    804d:	4c 89 e0             	mov    %r12,%rax
    8050:	41 5c                	pop    %r12
    8052:	c3                   	retq   
    8053:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8058:	83 ff 1e             	cmp    $0x1e,%edi
    805b:	7f ee                	jg     804b <galois_get_log_table+0x1b>
    805d:	e8 ae f2 ff ff       	callq  7310 <galois_create_log_tables.part.0>
    8062:	85 c0                	test   %eax,%eax
    8064:	75 e5                	jne    804b <galois_get_log_table+0x1b>
    8066:	4c 8b 24 eb          	mov    (%rbx,%rbp,8),%r12
    806a:	5b                   	pop    %rbx
    806b:	5d                   	pop    %rbp
    806c:	4c 89 e0             	mov    %r12,%rax
    806f:	41 5c                	pop    %r12
    8071:	c3                   	retq   
    8072:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    8079:	00 00 00 00 
    807d:	0f 1f 00             	nopl   (%rax)

0000000000008080 <galois_get_ilog_table>:
    8080:	f3 0f 1e fa          	endbr64 
    8084:	41 54                	push   %r12
    8086:	55                   	push   %rbp
    8087:	48 8d 2d 72 93 00 00 	lea    0x9372(%rip),%rbp        # 11400 <galois_ilog_tables>
    808e:	53                   	push   %rbx
    808f:	48 63 df             	movslq %edi,%rbx
    8092:	4c 8b 64 dd 00       	mov    0x0(%rbp,%rbx,8),%r12
    8097:	4d 85 e4             	test   %r12,%r12
    809a:	74 0c                	je     80a8 <galois_get_ilog_table+0x28>
    809c:	5b                   	pop    %rbx
    809d:	5d                   	pop    %rbp
    809e:	4c 89 e0             	mov    %r12,%rax
    80a1:	41 5c                	pop    %r12
    80a3:	c3                   	retq   
    80a4:	0f 1f 40 00          	nopl   0x0(%rax)
    80a8:	83 ff 1e             	cmp    $0x1e,%edi
    80ab:	7f ef                	jg     809c <galois_get_ilog_table+0x1c>
    80ad:	48 8d 05 6c 94 00 00 	lea    0x946c(%rip),%rax        # 11520 <galois_log_tables>
    80b4:	48 83 3c d8 00       	cmpq   $0x0,(%rax,%rbx,8)
    80b9:	75 e1                	jne    809c <galois_get_ilog_table+0x1c>
    80bb:	e8 50 f2 ff ff       	callq  7310 <galois_create_log_tables.part.0>
    80c0:	85 c0                	test   %eax,%eax
    80c2:	75 d8                	jne    809c <galois_get_ilog_table+0x1c>
    80c4:	4c 8b 64 dd 00       	mov    0x0(%rbp,%rbx,8),%r12
    80c9:	eb d1                	jmp    809c <galois_get_ilog_table+0x1c>
    80cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000080d0 <galois_region_xor>:
    80d0:	f3 0f 1e fa          	endbr64 
    80d4:	4c 63 c1             	movslq %ecx,%r8
    80d7:	49 01 f8             	add    %rdi,%r8
    80da:	4c 39 c7             	cmp    %r8,%rdi
    80dd:	73 28                	jae    8107 <galois_region_xor+0x37>
    80df:	48 89 f8             	mov    %rdi,%rax
    80e2:	48 f7 d0             	not    %rax
    80e5:	49 01 c0             	add    %rax,%r8
    80e8:	49 c1 e8 03          	shr    $0x3,%r8
    80ec:	31 c0                	xor    %eax,%eax
    80ee:	66 90                	xchg   %ax,%ax
    80f0:	48 8b 0c c7          	mov    (%rdi,%rax,8),%rcx
    80f4:	48 33 0c c6          	xor    (%rsi,%rax,8),%rcx
    80f8:	48 89 0c c2          	mov    %rcx,(%rdx,%rax,8)
    80fc:	48 89 c1             	mov    %rax,%rcx
    80ff:	48 ff c0             	inc    %rax
    8102:	49 39 c8             	cmp    %rcx,%r8
    8105:	75 e9                	jne    80f0 <galois_region_xor+0x20>
    8107:	c3                   	retq   
    8108:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    810f:	00 

0000000000008110 <galois_create_split_w8_tables>:
    8110:	f3 0f 1e fa          	endbr64 
    8114:	48 83 3d 64 90 00 00 	cmpq   $0x0,0x9064(%rip)        # 11180 <galois_split_w8>
    811b:	00 
    811c:	74 03                	je     8121 <galois_create_split_w8_tables+0x11>
    811e:	31 c0                	xor    %eax,%eax
    8120:	c3                   	retq   
    8121:	41 57                	push   %r15
    8123:	bf 08 00 00 00       	mov    $0x8,%edi
    8128:	41 56                	push   %r14
    812a:	41 55                	push   %r13
    812c:	41 54                	push   %r12
    812e:	55                   	push   %rbp
    812f:	53                   	push   %rbx
    8130:	48 83 ec 18          	sub    $0x18,%rsp
    8134:	e8 27 f4 ff ff       	callq  7560 <galois_create_mult_tables>
    8139:	85 c0                	test   %eax,%eax
    813b:	0f 88 00 01 00 00    	js     8241 <galois_create_split_w8_tables+0x131>
    8141:	31 db                	xor    %ebx,%ebx
    8143:	bf 00 00 04 00       	mov    $0x40000,%edi
    8148:	e8 b3 92 ff ff       	callq  1400 <malloc@plt>
    814d:	48 8d 0d 2c 90 00 00 	lea    0x902c(%rip),%rcx        # 11180 <galois_split_w8>
    8154:	48 89 04 d9          	mov    %rax,(%rcx,%rbx,8)
    8158:	89 dd                	mov    %ebx,%ebp
    815a:	48 85 c0             	test   %rax,%rax
    815d:	0f 84 bc 00 00 00    	je     821f <galois_create_split_w8_tables+0x10f>
    8163:	48 ff c3             	inc    %rbx
    8166:	48 83 fb 07          	cmp    $0x7,%rbx
    816a:	75 d7                	jne    8143 <galois_create_split_w8_tables+0x33>
    816c:	c7 44 24 04 00 00 00 	movl   $0x0,0x4(%rsp)
    8173:	00 
    8174:	8b 44 24 04          	mov    0x4(%rsp),%eax
    8178:	45 31 e4             	xor    %r12d,%r12d
    817b:	85 c0                	test   %eax,%eax
    817d:	41 0f 95 c4          	setne  %r12b
    8181:	44 8d 34 c5 00 00 00 	lea    0x0(,%rax,8),%r14d
    8188:	00 
    8189:	44 01 e0             	add    %r12d,%eax
    818c:	48 98                	cltq   
    818e:	48 8d 0d eb 8f 00 00 	lea    0x8feb(%rip),%rcx        # 11180 <galois_split_w8>
    8195:	48 8d 04 c1          	lea    (%rcx,%rax,8),%rax
    8199:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    819e:	41 c1 e4 03          	shl    $0x3,%r12d
    81a2:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    81a7:	45 31 ed             	xor    %r13d,%r13d
    81aa:	48 8b 18             	mov    (%rax),%rbx
    81ad:	45 31 ff             	xor    %r15d,%r15d
    81b0:	c4 c2 09 f7 ed       	shlx   %r14d,%r13d,%ebp
    81b5:	0f 1f 00             	nopl   (%rax)
    81b8:	ba 20 00 00 00       	mov    $0x20,%edx
    81bd:	89 ef                	mov    %ebp,%edi
    81bf:	c4 c2 19 f7 f7       	shlx   %r12d,%r15d,%esi
    81c4:	e8 97 f6 ff ff       	callq  7860 <galois_shift_multiply>
    81c9:	42 89 04 bb          	mov    %eax,(%rbx,%r15,4)
    81cd:	49 ff c7             	inc    %r15
    81d0:	49 81 ff 00 01 00 00 	cmp    $0x100,%r15
    81d7:	75 df                	jne    81b8 <galois_create_split_w8_tables+0xa8>
    81d9:	41 ff c5             	inc    %r13d
    81dc:	48 81 c3 00 04 00 00 	add    $0x400,%rbx
    81e3:	41 81 fd 00 01 00 00 	cmp    $0x100,%r13d
    81ea:	75 c1                	jne    81ad <galois_create_split_w8_tables+0x9d>
    81ec:	41 83 c4 08          	add    $0x8,%r12d
    81f0:	48 83 44 24 08 08    	addq   $0x8,0x8(%rsp)
    81f6:	41 83 fc 20          	cmp    $0x20,%r12d
    81fa:	75 a6                	jne    81a2 <galois_create_split_w8_tables+0x92>
    81fc:	83 44 24 04 03       	addl   $0x3,0x4(%rsp)
    8201:	8b 44 24 04          	mov    0x4(%rsp),%eax
    8205:	83 f8 06             	cmp    $0x6,%eax
    8208:	0f 85 66 ff ff ff    	jne    8174 <galois_create_split_w8_tables+0x64>
    820e:	31 c0                	xor    %eax,%eax
    8210:	48 83 c4 18          	add    $0x18,%rsp
    8214:	5b                   	pop    %rbx
    8215:	5d                   	pop    %rbp
    8216:	41 5c                	pop    %r12
    8218:	41 5d                	pop    %r13
    821a:	41 5e                	pop    %r14
    821c:	41 5f                	pop    %r15
    821e:	c3                   	retq   
    821f:	8d 5b ff             	lea    -0x1(%rbx),%ebx
    8222:	85 ed                	test   %ebp,%ebp
    8224:	74 1b                	je     8241 <galois_create_split_w8_tables+0x131>
    8226:	48 63 db             	movslq %ebx,%rbx
    8229:	48 8d 05 50 8f 00 00 	lea    0x8f50(%rip),%rax        # 11180 <galois_split_w8>
    8230:	48 8b 3c d8          	mov    (%rax,%rbx,8),%rdi
    8234:	48 ff cb             	dec    %rbx
    8237:	e8 44 90 ff ff       	callq  1280 <free@plt>
    823c:	83 fb ff             	cmp    $0xffffffff,%ebx
    823f:	75 e8                	jne    8229 <galois_create_split_w8_tables+0x119>
    8241:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    8246:	eb c8                	jmp    8210 <galois_create_split_w8_tables+0x100>
    8248:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    824f:	00 

0000000000008250 <galois_w32_region_multiply>:
    8250:	f3 0f 1e fa          	endbr64 
    8254:	41 56                	push   %r14
    8256:	48 63 d2             	movslq %edx,%rdx
    8259:	41 55                	push   %r13
    825b:	41 54                	push   %r12
    825d:	55                   	push   %rbp
    825e:	48 89 cd             	mov    %rcx,%rbp
    8261:	53                   	push   %rbx
    8262:	48 89 fb             	mov    %rdi,%rbx
    8265:	48 83 ec 30          	sub    $0x30,%rsp
    8269:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    8270:	00 00 
    8272:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    8277:	31 c0                	xor    %eax,%eax
    8279:	48 85 c9             	test   %rcx,%rcx
    827c:	48 0f 44 ef          	cmove  %rdi,%rbp
    8280:	48 c1 ea 02          	shr    $0x2,%rdx
    8284:	48 83 3d f4 8e 00 00 	cmpq   $0x0,0x8ef4(%rip)        # 11180 <galois_split_w8>
    828b:	00 
    828c:	49 89 d6             	mov    %rdx,%r14
    828f:	0f 84 28 01 00 00    	je     83bd <galois_w32_region_multiply+0x16d>
    8295:	48 8d 4c 24 10       	lea    0x10(%rsp),%rcx
    829a:	48 89 cf             	mov    %rcx,%rdi
    829d:	31 d2                	xor    %edx,%edx
    829f:	c4 e2 6a f7 c6       	sarx   %edx,%esi,%eax
    82a4:	c1 e0 08             	shl    $0x8,%eax
    82a7:	25 ff ff 00 00       	and    $0xffff,%eax
    82ac:	83 c2 08             	add    $0x8,%edx
    82af:	89 07                	mov    %eax,(%rdi)
    82b1:	48 83 c7 04          	add    $0x4,%rdi
    82b5:	83 fa 20             	cmp    $0x20,%edx
    82b8:	75 e5                	jne    829f <galois_w32_region_multiply+0x4f>
    82ba:	45 85 c0             	test   %r8d,%r8d
    82bd:	75 6b                	jne    832a <galois_w32_region_multiply+0xda>
    82bf:	45 85 f6             	test   %r14d,%r14d
    82c2:	0f 8e d8 00 00 00    	jle    83a0 <galois_w32_region_multiply+0x150>
    82c8:	45 8d 6e ff          	lea    -0x1(%r14),%r13d
    82cc:	45 31 e4             	xor    %r12d,%r12d
    82cf:	90                   	nop
    82d0:	46 8b 14 a3          	mov    (%rbx,%r12,4),%r10d
    82d4:	4c 8d 1d a5 8e 00 00 	lea    0x8ea5(%rip),%r11        # 11180 <galois_split_w8>
    82db:	31 ff                	xor    %edi,%edi
    82dd:	45 31 c9             	xor    %r9d,%r9d
    82e0:	46 8b 04 89          	mov    (%rcx,%r9,4),%r8d
    82e4:	4c 89 de             	mov    %r11,%rsi
    82e7:	31 d2                	xor    %edx,%edx
    82e9:	c4 c2 6b f7 c2       	shrx   %edx,%r10d,%eax
    82ee:	0f b6 c0             	movzbl %al,%eax
    82f1:	4c 8b 36             	mov    (%rsi),%r14
    82f4:	44 09 c0             	or     %r8d,%eax
    82f7:	48 98                	cltq   
    82f9:	83 c2 08             	add    $0x8,%edx
    82fc:	41 33 3c 86          	xor    (%r14,%rax,4),%edi
    8300:	48 83 c6 08          	add    $0x8,%rsi
    8304:	83 fa 20             	cmp    $0x20,%edx
    8307:	75 e0                	jne    82e9 <galois_w32_region_multiply+0x99>
    8309:	49 ff c1             	inc    %r9
    830c:	49 83 c3 08          	add    $0x8,%r11
    8310:	49 83 f9 04          	cmp    $0x4,%r9
    8314:	75 ca                	jne    82e0 <galois_w32_region_multiply+0x90>
    8316:	42 89 7c a5 00       	mov    %edi,0x0(%rbp,%r12,4)
    831b:	49 8d 44 24 01       	lea    0x1(%r12),%rax
    8320:	4d 39 e5             	cmp    %r12,%r13
    8323:	74 7b                	je     83a0 <galois_w32_region_multiply+0x150>
    8325:	49 89 c4             	mov    %rax,%r12
    8328:	eb a6                	jmp    82d0 <galois_w32_region_multiply+0x80>
    832a:	45 8d 6e ff          	lea    -0x1(%r14),%r13d
    832e:	45 31 e4             	xor    %r12d,%r12d
    8331:	45 85 f6             	test   %r14d,%r14d
    8334:	7e 6a                	jle    83a0 <galois_w32_region_multiply+0x150>
    8336:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    833d:	00 00 00 
    8340:	46 8b 14 a3          	mov    (%rbx,%r12,4),%r10d
    8344:	4c 8d 1d 35 8e 00 00 	lea    0x8e35(%rip),%r11        # 11180 <galois_split_w8>
    834b:	45 31 c9             	xor    %r9d,%r9d
    834e:	31 ff                	xor    %edi,%edi
    8350:	46 8b 04 89          	mov    (%rcx,%r9,4),%r8d
    8354:	4c 89 de             	mov    %r11,%rsi
    8357:	31 d2                	xor    %edx,%edx
    8359:	c4 c2 6b f7 c2       	shrx   %edx,%r10d,%eax
    835e:	0f b6 c0             	movzbl %al,%eax
    8361:	4c 8b 36             	mov    (%rsi),%r14
    8364:	44 09 c0             	or     %r8d,%eax
    8367:	48 98                	cltq   
    8369:	83 c2 08             	add    $0x8,%edx
    836c:	41 33 3c 86          	xor    (%r14,%rax,4),%edi
    8370:	48 83 c6 08          	add    $0x8,%rsi
    8374:	83 fa 20             	cmp    $0x20,%edx
    8377:	75 e0                	jne    8359 <galois_w32_region_multiply+0x109>
    8379:	49 ff c1             	inc    %r9
    837c:	49 83 c3 08          	add    $0x8,%r11
    8380:	49 83 f9 04          	cmp    $0x4,%r9
    8384:	75 ca                	jne    8350 <galois_w32_region_multiply+0x100>
    8386:	42 31 7c a5 00       	xor    %edi,0x0(%rbp,%r12,4)
    838b:	49 8d 44 24 01       	lea    0x1(%r12),%rax
    8390:	4d 39 e5             	cmp    %r12,%r13
    8393:	74 0b                	je     83a0 <galois_w32_region_multiply+0x150>
    8395:	49 89 c4             	mov    %rax,%r12
    8398:	eb a6                	jmp    8340 <galois_w32_region_multiply+0xf0>
    839a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    83a0:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    83a5:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    83ac:	00 00 
    83ae:	75 58                	jne    8408 <galois_w32_region_multiply+0x1b8>
    83b0:	48 83 c4 30          	add    $0x30,%rsp
    83b4:	5b                   	pop    %rbx
    83b5:	5d                   	pop    %rbp
    83b6:	41 5c                	pop    %r12
    83b8:	41 5d                	pop    %r13
    83ba:	41 5e                	pop    %r14
    83bc:	c3                   	retq   
    83bd:	bf 08 00 00 00       	mov    $0x8,%edi
    83c2:	44 89 44 24 0c       	mov    %r8d,0xc(%rsp)
    83c7:	89 74 24 08          	mov    %esi,0x8(%rsp)
    83cb:	e8 40 fd ff ff       	callq  8110 <galois_create_split_w8_tables>
    83d0:	85 c0                	test   %eax,%eax
    83d2:	8b 74 24 08          	mov    0x8(%rsp),%esi
    83d6:	44 8b 44 24 0c       	mov    0xc(%rsp),%r8d
    83db:	0f 89 b4 fe ff ff    	jns    8295 <galois_w32_region_multiply+0x45>
    83e1:	48 8b 0d 58 8d 00 00 	mov    0x8d58(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    83e8:	48 8d 3d f1 25 00 00 	lea    0x25f1(%rip),%rdi        # a9e0 <__PRETTY_FUNCTION__.5741+0x307>
    83ef:	ba 47 00 00 00       	mov    $0x47,%edx
    83f4:	be 01 00 00 00       	mov    $0x1,%esi
    83f9:	e8 72 90 ff ff       	callq  1470 <fwrite@plt>
    83fe:	bf 01 00 00 00       	mov    $0x1,%edi
    8403:	e8 58 90 ff ff       	callq  1460 <exit@plt>
    8408:	e8 03 8f ff ff       	callq  1310 <__stack_chk_fail@plt>
    840d:	0f 1f 00             	nopl   (%rax)

0000000000008410 <galois_split_w8_multiply>:
    8410:	f3 0f 1e fa          	endbr64 
    8414:	53                   	push   %rbx
    8415:	4c 8d 1d 64 8d 00 00 	lea    0x8d64(%rip),%r11        # 11180 <galois_split_w8>
    841c:	45 31 c9             	xor    %r9d,%r9d
    841f:	45 31 d2             	xor    %r10d,%r10d
    8422:	46 8d 04 d5 00 00 00 	lea    0x0(,%r10,8),%r8d
    8429:	00 
    842a:	c4 62 3a f7 c7       	sarx   %r8d,%edi,%r8d
    842f:	41 c1 e0 08          	shl    $0x8,%r8d
    8433:	45 0f b7 c0          	movzwl %r8w,%r8d
    8437:	4c 89 d9             	mov    %r11,%rcx
    843a:	31 d2                	xor    %edx,%edx
    843c:	c4 e2 6a f7 c6       	sarx   %edx,%esi,%eax
    8441:	0f b6 c0             	movzbl %al,%eax
    8444:	48 8b 19             	mov    (%rcx),%rbx
    8447:	44 09 c0             	or     %r8d,%eax
    844a:	48 98                	cltq   
    844c:	83 c2 08             	add    $0x8,%edx
    844f:	44 33 0c 83          	xor    (%rbx,%rax,4),%r9d
    8453:	48 83 c1 08          	add    $0x8,%rcx
    8457:	83 fa 20             	cmp    $0x20,%edx
    845a:	75 e0                	jne    843c <galois_split_w8_multiply+0x2c>
    845c:	41 ff c2             	inc    %r10d
    845f:	49 83 c3 08          	add    $0x8,%r11
    8463:	41 83 fa 04          	cmp    $0x4,%r10d
    8467:	75 b9                	jne    8422 <galois_split_w8_multiply+0x12>
    8469:	44 89 c8             	mov    %r9d,%eax
    846c:	5b                   	pop    %rbx
    846d:	c3                   	retq   
    846e:	66 90                	xchg   %ax,%ax

0000000000008470 <galois_single_multiply.part.0>:
    8470:	41 55                	push   %r13
    8472:	48 8d 05 c7 27 00 00 	lea    0x27c7(%rip),%rax        # ac40 <mult_type>
    8479:	41 54                	push   %r12
    847b:	41 89 fc             	mov    %edi,%r12d
    847e:	53                   	push   %rbx
    847f:	48 63 da             	movslq %edx,%rbx
    8482:	48 83 ec 10          	sub    $0x10,%rsp
    8486:	8b 04 98             	mov    (%rax,%rbx,4),%eax
    8489:	83 f8 0b             	cmp    $0xb,%eax
    848c:	74 62                	je     84f0 <galois_single_multiply.part.0+0x80>
    848e:	83 f8 0d             	cmp    $0xd,%eax
    8491:	74 25                	je     84b8 <galois_single_multiply.part.0+0x48>
    8493:	83 f8 0e             	cmp    $0xe,%eax
    8496:	0f 84 bc 00 00 00    	je     8558 <galois_single_multiply.part.0+0xe8>
    849c:	83 f8 0c             	cmp    $0xc,%eax
    849f:	0f 85 47 01 00 00    	jne    85ec <galois_single_multiply.part.0+0x17c>
    84a5:	48 83 c4 10          	add    $0x10,%rsp
    84a9:	5b                   	pop    %rbx
    84aa:	41 5c                	pop    %r12
    84ac:	41 5d                	pop    %r13
    84ae:	e9 ad f3 ff ff       	jmpq   7860 <galois_shift_multiply>
    84b3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    84b8:	4c 8d 2d 61 90 00 00 	lea    0x9061(%rip),%r13        # 11520 <galois_log_tables>
    84bf:	49 8b 4c dd 00       	mov    0x0(%r13,%rbx,8),%rcx
    84c4:	48 85 c9             	test   %rcx,%rcx
    84c7:	74 57                	je     8520 <galois_single_multiply.part.0+0xb0>
    84c9:	48 63 f6             	movslq %esi,%rsi
    84cc:	8b 04 b1             	mov    (%rcx,%rsi,4),%eax
    84cf:	49 63 fc             	movslq %r12d,%rdi
    84d2:	48 8d 15 27 8f 00 00 	lea    0x8f27(%rip),%rdx        # 11400 <galois_ilog_tables>
    84d9:	48 8b 14 da          	mov    (%rdx,%rbx,8),%rdx
    84dd:	03 04 b9             	add    (%rcx,%rdi,4),%eax
    84e0:	48 98                	cltq   
    84e2:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    84e5:	48 83 c4 10          	add    $0x10,%rsp
    84e9:	5b                   	pop    %rbx
    84ea:	41 5c                	pop    %r12
    84ec:	41 5d                	pop    %r13
    84ee:	c3                   	retq   
    84ef:	90                   	nop
    84f0:	4c 8d 2d e9 8d 00 00 	lea    0x8de9(%rip),%r13        # 112e0 <galois_mult_tables>
    84f7:	49 8b 44 dd 00       	mov    0x0(%r13,%rbx,8),%rax
    84fc:	48 85 c0             	test   %rax,%rax
    84ff:	74 77                	je     8578 <galois_single_multiply.part.0+0x108>
    8501:	c4 42 69 f7 e4       	shlx   %edx,%r12d,%r12d
    8506:	41 09 f4             	or     %esi,%r12d
    8509:	4d 63 e4             	movslq %r12d,%r12
    850c:	42 8b 04 a0          	mov    (%rax,%r12,4),%eax
    8510:	48 83 c4 10          	add    $0x10,%rsp
    8514:	5b                   	pop    %rbx
    8515:	41 5c                	pop    %r12
    8517:	41 5d                	pop    %r13
    8519:	c3                   	retq   
    851a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    8520:	89 74 24 08          	mov    %esi,0x8(%rsp)
    8524:	83 fa 1e             	cmp    $0x1e,%edx
    8527:	0f 8f b4 00 00 00    	jg     85e1 <galois_single_multiply.part.0+0x171>
    852d:	89 d7                	mov    %edx,%edi
    852f:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    8533:	e8 d8 ed ff ff       	callq  7310 <galois_create_log_tables.part.0>
    8538:	85 c0                	test   %eax,%eax
    853a:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    853e:	0f 88 9d 00 00 00    	js     85e1 <galois_single_multiply.part.0+0x171>
    8544:	49 8b 4c dd 00       	mov    0x0(%r13,%rbx,8),%rcx
    8549:	8b 74 24 08          	mov    0x8(%rsp),%esi
    854d:	e9 77 ff ff ff       	jmpq   84c9 <galois_single_multiply.part.0+0x59>
    8552:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    8558:	48 83 3d 20 8c 00 00 	cmpq   $0x0,0x8c20(%rip)        # 11180 <galois_split_w8>
    855f:	00 
    8560:	74 3e                	je     85a0 <galois_single_multiply.part.0+0x130>
    8562:	48 83 c4 10          	add    $0x10,%rsp
    8566:	5b                   	pop    %rbx
    8567:	44 89 e7             	mov    %r12d,%edi
    856a:	41 5c                	pop    %r12
    856c:	41 5d                	pop    %r13
    856e:	e9 9d fe ff ff       	jmpq   8410 <galois_split_w8_multiply>
    8573:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8578:	89 d7                	mov    %edx,%edi
    857a:	89 54 24 08          	mov    %edx,0x8(%rsp)
    857e:	89 74 24 0c          	mov    %esi,0xc(%rsp)
    8582:	e8 d9 ef ff ff       	callq  7560 <galois_create_mult_tables>
    8587:	85 c0                	test   %eax,%eax
    8589:	8b 54 24 08          	mov    0x8(%rsp),%edx
    858d:	78 68                	js     85f7 <galois_single_multiply.part.0+0x187>
    858f:	49 8b 44 dd 00       	mov    0x0(%r13,%rbx,8),%rax
    8594:	8b 74 24 0c          	mov    0xc(%rsp),%esi
    8598:	e9 64 ff ff ff       	jmpq   8501 <galois_single_multiply.part.0+0x91>
    859d:	0f 1f 00             	nopl   (%rax)
    85a0:	31 c0                	xor    %eax,%eax
    85a2:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    85a6:	89 74 24 08          	mov    %esi,0x8(%rsp)
    85aa:	e8 61 fb ff ff       	callq  8110 <galois_create_split_w8_tables>
    85af:	85 c0                	test   %eax,%eax
    85b1:	8b 74 24 08          	mov    0x8(%rsp),%esi
    85b5:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    85b9:	79 a7                	jns    8562 <galois_single_multiply.part.0+0xf2>
    85bb:	89 d1                	mov    %edx,%ecx
    85bd:	48 8d 15 cc 24 00 00 	lea    0x24cc(%rip),%rdx        # aa90 <__PRETTY_FUNCTION__.5741+0x3b7>
    85c4:	48 8b 3d 75 8b 00 00 	mov    0x8b75(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    85cb:	be 01 00 00 00       	mov    $0x1,%esi
    85d0:	31 c0                	xor    %eax,%eax
    85d2:	e8 a9 8e ff ff       	callq  1480 <__fprintf_chk@plt>
    85d7:	bf 01 00 00 00       	mov    $0x1,%edi
    85dc:	e8 7f 8e ff ff       	callq  1460 <exit@plt>
    85e1:	89 d1                	mov    %edx,%ecx
    85e3:	48 8d 15 76 24 00 00 	lea    0x2476(%rip),%rdx        # aa60 <__PRETTY_FUNCTION__.5741+0x387>
    85ea:	eb d8                	jmp    85c4 <galois_single_multiply.part.0+0x154>
    85ec:	89 d1                	mov    %edx,%ecx
    85ee:	48 8d 15 d3 24 00 00 	lea    0x24d3(%rip),%rdx        # aac8 <__PRETTY_FUNCTION__.5741+0x3ef>
    85f5:	eb cd                	jmp    85c4 <galois_single_multiply.part.0+0x154>
    85f7:	89 d1                	mov    %edx,%ecx
    85f9:	48 8d 15 28 24 00 00 	lea    0x2428(%rip),%rdx        # aa28 <__PRETTY_FUNCTION__.5741+0x34f>
    8600:	eb c2                	jmp    85c4 <galois_single_multiply.part.0+0x154>
    8602:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    8609:	00 00 00 00 
    860d:	0f 1f 00             	nopl   (%rax)

0000000000008610 <galois_single_multiply>:
    8610:	f3 0f 1e fa          	endbr64 
    8614:	85 ff                	test   %edi,%edi
    8616:	74 10                	je     8628 <galois_single_multiply+0x18>
    8618:	85 f6                	test   %esi,%esi
    861a:	74 0c                	je     8628 <galois_single_multiply+0x18>
    861c:	e9 4f fe ff ff       	jmpq   8470 <galois_single_multiply.part.0>
    8621:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8628:	31 c0                	xor    %eax,%eax
    862a:	c3                   	retq   
    862b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000008630 <galois_single_divide>:
    8630:	f3 0f 1e fa          	endbr64 
    8634:	41 56                	push   %r14
    8636:	48 8d 05 03 26 00 00 	lea    0x2603(%rip),%rax        # ac40 <mult_type>
    863d:	41 55                	push   %r13
    863f:	41 89 fd             	mov    %edi,%r13d
    8642:	41 54                	push   %r12
    8644:	55                   	push   %rbp
    8645:	48 63 ee             	movslq %esi,%rbp
    8648:	53                   	push   %rbx
    8649:	48 63 da             	movslq %edx,%rbx
    864c:	48 83 ec 10          	sub    $0x10,%rsp
    8650:	8b 04 98             	mov    (%rax,%rbx,4),%eax
    8653:	83 f8 0b             	cmp    $0xb,%eax
    8656:	74 78                	je     86d0 <galois_single_divide+0xa0>
    8658:	83 f8 0d             	cmp    $0xd,%eax
    865b:	74 23                	je     8680 <galois_single_divide+0x50>
    865d:	85 ed                	test   %ebp,%ebp
    865f:	0f 84 0c 01 00 00    	je     8771 <galois_single_divide+0x141>
    8665:	85 ff                	test   %edi,%edi
    8667:	0f 85 93 00 00 00    	jne    8700 <galois_single_divide+0xd0>
    866d:	31 c0                	xor    %eax,%eax
    866f:	48 83 c4 10          	add    $0x10,%rsp
    8673:	5b                   	pop    %rbx
    8674:	5d                   	pop    %rbp
    8675:	41 5c                	pop    %r12
    8677:	41 5d                	pop    %r13
    8679:	41 5e                	pop    %r14
    867b:	c3                   	retq   
    867c:	0f 1f 40 00          	nopl   0x0(%rax)
    8680:	85 ed                	test   %ebp,%ebp
    8682:	0f 84 f3 00 00 00    	je     877b <galois_single_divide+0x14b>
    8688:	85 ff                	test   %edi,%edi
    868a:	74 e1                	je     866d <galois_single_divide+0x3d>
    868c:	4c 8d 35 8d 8e 00 00 	lea    0x8e8d(%rip),%r14        # 11520 <galois_log_tables>
    8693:	49 8b 0c de          	mov    (%r14,%rbx,8),%rcx
    8697:	48 85 c9             	test   %rcx,%rcx
    869a:	0f 84 b0 00 00 00    	je     8750 <galois_single_divide+0x120>
    86a0:	4d 63 e5             	movslq %r13d,%r12
    86a3:	42 8b 04 a1          	mov    (%rcx,%r12,4),%eax
    86a7:	48 8d 15 52 8d 00 00 	lea    0x8d52(%rip),%rdx        # 11400 <galois_ilog_tables>
    86ae:	48 8b 14 da          	mov    (%rdx,%rbx,8),%rdx
    86b2:	2b 04 a9             	sub    (%rcx,%rbp,4),%eax
    86b5:	48 98                	cltq   
    86b7:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    86ba:	48 83 c4 10          	add    $0x10,%rsp
    86be:	5b                   	pop    %rbx
    86bf:	5d                   	pop    %rbp
    86c0:	41 5c                	pop    %r12
    86c2:	41 5d                	pop    %r13
    86c4:	41 5e                	pop    %r14
    86c6:	c3                   	retq   
    86c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    86ce:	00 00 
    86d0:	4c 8d 35 e9 8a 00 00 	lea    0x8ae9(%rip),%r14        # 111c0 <galois_div_tables>
    86d7:	49 8b 04 de          	mov    (%r14,%rbx,8),%rax
    86db:	48 85 c0             	test   %rax,%rax
    86de:	74 50                	je     8730 <galois_single_divide+0x100>
    86e0:	c4 42 69 f7 e5       	shlx   %edx,%r13d,%r12d
    86e5:	41 09 ec             	or     %ebp,%r12d
    86e8:	4d 63 e4             	movslq %r12d,%r12
    86eb:	42 8b 04 a0          	mov    (%rax,%r12,4),%eax
    86ef:	48 83 c4 10          	add    $0x10,%rsp
    86f3:	5b                   	pop    %rbx
    86f4:	5d                   	pop    %rbp
    86f5:	41 5c                	pop    %r12
    86f7:	41 5d                	pop    %r13
    86f9:	41 5e                	pop    %r14
    86fb:	c3                   	retq   
    86fc:	0f 1f 40 00          	nopl   0x0(%rax)
    8700:	89 d6                	mov    %edx,%esi
    8702:	89 ef                	mov    %ebp,%edi
    8704:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    8708:	e8 b3 00 00 00       	callq  87c0 <galois_inverse>
    870d:	89 c6                	mov    %eax,%esi
    870f:	85 c0                	test   %eax,%eax
    8711:	0f 84 56 ff ff ff    	je     866d <galois_single_divide+0x3d>
    8717:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    871b:	48 83 c4 10          	add    $0x10,%rsp
    871f:	5b                   	pop    %rbx
    8720:	5d                   	pop    %rbp
    8721:	41 5c                	pop    %r12
    8723:	44 89 ef             	mov    %r13d,%edi
    8726:	41 5d                	pop    %r13
    8728:	41 5e                	pop    %r14
    872a:	e9 41 fd ff ff       	jmpq   8470 <galois_single_multiply.part.0>
    872f:	90                   	nop
    8730:	89 d7                	mov    %edx,%edi
    8732:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    8736:	e8 25 ee ff ff       	callq  7560 <galois_create_mult_tables>
    873b:	85 c0                	test   %eax,%eax
    873d:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    8741:	78 68                	js     87ab <galois_single_divide+0x17b>
    8743:	49 8b 04 de          	mov    (%r14,%rbx,8),%rax
    8747:	eb 97                	jmp    86e0 <galois_single_divide+0xb0>
    8749:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8750:	83 fa 1e             	cmp    $0x1e,%edx
    8753:	7f 30                	jg     8785 <galois_single_divide+0x155>
    8755:	89 d7                	mov    %edx,%edi
    8757:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    875b:	e8 b0 eb ff ff       	callq  7310 <galois_create_log_tables.part.0>
    8760:	85 c0                	test   %eax,%eax
    8762:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    8766:	78 1d                	js     8785 <galois_single_divide+0x155>
    8768:	49 8b 0c de          	mov    (%r14,%rbx,8),%rcx
    876c:	e9 2f ff ff ff       	jmpq   86a0 <galois_single_divide+0x70>
    8771:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    8776:	e9 f4 fe ff ff       	jmpq   866f <galois_single_divide+0x3f>
    877b:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    8780:	e9 ea fe ff ff       	jmpq   866f <galois_single_divide+0x3f>
    8785:	89 d1                	mov    %edx,%ecx
    8787:	48 8d 15 d2 22 00 00 	lea    0x22d2(%rip),%rdx        # aa60 <__PRETTY_FUNCTION__.5741+0x387>
    878e:	48 8b 3d ab 89 00 00 	mov    0x89ab(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    8795:	be 01 00 00 00       	mov    $0x1,%esi
    879a:	31 c0                	xor    %eax,%eax
    879c:	e8 df 8c ff ff       	callq  1480 <__fprintf_chk@plt>
    87a1:	bf 01 00 00 00       	mov    $0x1,%edi
    87a6:	e8 b5 8c ff ff       	callq  1460 <exit@plt>
    87ab:	89 d1                	mov    %edx,%ecx
    87ad:	48 8d 15 74 22 00 00 	lea    0x2274(%rip),%rdx        # aa28 <__PRETTY_FUNCTION__.5741+0x34f>
    87b4:	eb d8                	jmp    878e <galois_single_divide+0x15e>
    87b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    87bd:	00 00 00 

00000000000087c0 <galois_inverse>:
    87c0:	f3 0f 1e fa          	endbr64 
    87c4:	89 f2                	mov    %esi,%edx
    87c6:	85 ff                	test   %edi,%edi
    87c8:	74 2e                	je     87f8 <galois_inverse+0x38>
    87ca:	48 63 c6             	movslq %esi,%rax
    87cd:	48 8d 0d 6c 24 00 00 	lea    0x246c(%rip),%rcx        # ac40 <mult_type>
    87d4:	8b 04 81             	mov    (%rcx,%rax,4),%eax
    87d7:	83 e0 fd             	and    $0xfffffffd,%eax
    87da:	83 f8 0c             	cmp    $0xc,%eax
    87dd:	74 11                	je     87f0 <galois_inverse+0x30>
    87df:	89 fe                	mov    %edi,%esi
    87e1:	bf 01 00 00 00       	mov    $0x1,%edi
    87e6:	e9 45 fe ff ff       	jmpq   8630 <galois_single_divide>
    87eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    87f0:	e9 8b f6 ff ff       	jmpq   7e80 <galois_shift_inverse>
    87f5:	0f 1f 00             	nopl   (%rax)
    87f8:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    87fd:	c3                   	retq   
    87fe:	66 90                	xchg   %ax,%ax

0000000000008800 <reed_sol_r6_coding_matrix>:
    8800:	f3 0f 1e fa          	endbr64 
    8804:	41 55                	push   %r13
    8806:	8d 46 f8             	lea    -0x8(%rsi),%eax
    8809:	4c 63 ef             	movslq %edi,%r13
    880c:	41 54                	push   %r12
    880e:	55                   	push   %rbp
    880f:	89 f5                	mov    %esi,%ebp
    8811:	53                   	push   %rbx
    8812:	48 83 ec 08          	sub    $0x8,%rsp
    8816:	83 e0 f7             	and    $0xfffffff7,%eax
    8819:	74 09                	je     8824 <reed_sol_r6_coding_matrix+0x24>
    881b:	83 fe 20             	cmp    $0x20,%esi
    881e:	0f 85 8c 00 00 00    	jne    88b0 <reed_sol_r6_coding_matrix+0xb0>
    8824:	43 8d 7c 2d 00       	lea    0x0(%r13,%r13,1),%edi
    8829:	48 63 ff             	movslq %edi,%rdi
    882c:	48 c1 e7 02          	shl    $0x2,%rdi
    8830:	e8 cb 8b ff ff       	callq  1400 <malloc@plt>
    8835:	49 89 c4             	mov    %rax,%r12
    8838:	48 85 c0             	test   %rax,%rax
    883b:	74 73                	je     88b0 <reed_sol_r6_coding_matrix+0xb0>
    883d:	45 85 ed             	test   %r13d,%r13d
    8840:	0f 8e 82 00 00 00    	jle    88c8 <reed_sol_r6_coding_matrix+0xc8>
    8846:	41 8d 55 ff          	lea    -0x1(%r13),%edx
    884a:	48 8d 54 90 04       	lea    0x4(%rax,%rdx,4),%rdx
    884f:	90                   	nop
    8850:	c7 00 01 00 00 00    	movl   $0x1,(%rax)
    8856:	48 83 c0 04          	add    $0x4,%rax
    885a:	48 39 d0             	cmp    %rdx,%rax
    885d:	75 f1                	jne    8850 <reed_sol_r6_coding_matrix+0x50>
    885f:	49 63 c5             	movslq %r13d,%rax
    8862:	49 8d 1c 84          	lea    (%r12,%rax,4),%rbx
    8866:	c7 03 01 00 00 00    	movl   $0x1,(%rbx)
    886c:	41 83 fd 01          	cmp    $0x1,%r13d
    8870:	7e 30                	jle    88a2 <reed_sol_r6_coding_matrix+0xa2>
    8872:	41 8d 55 fe          	lea    -0x2(%r13),%edx
    8876:	48 8d 44 10 01       	lea    0x1(%rax,%rdx,1),%rax
    887b:	4d 8d 2c 84          	lea    (%r12,%rax,4),%r13
    887f:	bf 01 00 00 00       	mov    $0x1,%edi
    8884:	0f 1f 40 00          	nopl   0x0(%rax)
    8888:	89 ea                	mov    %ebp,%edx
    888a:	be 02 00 00 00       	mov    $0x2,%esi
    888f:	e8 7c fd ff ff       	callq  8610 <galois_single_multiply>
    8894:	89 43 04             	mov    %eax,0x4(%rbx)
    8897:	48 83 c3 04          	add    $0x4,%rbx
    889b:	89 c7                	mov    %eax,%edi
    889d:	4c 39 eb             	cmp    %r13,%rbx
    88a0:	75 e6                	jne    8888 <reed_sol_r6_coding_matrix+0x88>
    88a2:	48 83 c4 08          	add    $0x8,%rsp
    88a6:	5b                   	pop    %rbx
    88a7:	5d                   	pop    %rbp
    88a8:	4c 89 e0             	mov    %r12,%rax
    88ab:	41 5c                	pop    %r12
    88ad:	41 5d                	pop    %r13
    88af:	c3                   	retq   
    88b0:	48 83 c4 08          	add    $0x8,%rsp
    88b4:	5b                   	pop    %rbx
    88b5:	45 31 e4             	xor    %r12d,%r12d
    88b8:	5d                   	pop    %rbp
    88b9:	4c 89 e0             	mov    %r12,%rax
    88bc:	41 5c                	pop    %r12
    88be:	41 5d                	pop    %r13
    88c0:	c3                   	retq   
    88c1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    88c8:	42 c7 04 a8 01 00 00 	movl   $0x1,(%rax,%r13,4)
    88cf:	00 
    88d0:	eb d0                	jmp    88a2 <reed_sol_r6_coding_matrix+0xa2>
    88d2:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    88d9:	00 00 00 00 
    88dd:	0f 1f 00             	nopl   (%rax)

00000000000088e0 <reed_sol_galois_w32_region_multby_2>:
    88e0:	f3 0f 1e fa          	endbr64 
    88e4:	55                   	push   %rbp
    88e5:	89 f5                	mov    %esi,%ebp
    88e7:	53                   	push   %rbx
    88e8:	48 89 fb             	mov    %rdi,%rbx
    88eb:	48 83 ec 08          	sub    $0x8,%rsp
    88ef:	83 3d 92 57 00 00 ff 	cmpl   $0xffffffff,0x5792(%rip)        # e088 <prim32>
    88f6:	74 40                	je     8938 <reed_sol_galois_w32_region_multby_2+0x58>
    88f8:	48 63 f5             	movslq %ebp,%rsi
    88fb:	48 01 de             	add    %rbx,%rsi
    88fe:	8b 3d 84 57 00 00    	mov    0x5784(%rip),%edi        # e088 <prim32>
    8904:	48 39 f3             	cmp    %rsi,%rbx
    8907:	73 21                	jae    892a <reed_sol_galois_w32_region_multby_2+0x4a>
    8909:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8910:	8b 13                	mov    (%rbx),%edx
    8912:	8d 04 12             	lea    (%rdx,%rdx,1),%eax
    8915:	89 c1                	mov    %eax,%ecx
    8917:	31 f9                	xor    %edi,%ecx
    8919:	85 d2                	test   %edx,%edx
    891b:	0f 48 c1             	cmovs  %ecx,%eax
    891e:	48 83 c3 04          	add    $0x4,%rbx
    8922:	89 43 fc             	mov    %eax,-0x4(%rbx)
    8925:	48 39 de             	cmp    %rbx,%rsi
    8928:	77 e6                	ja     8910 <reed_sol_galois_w32_region_multby_2+0x30>
    892a:	48 83 c4 08          	add    $0x8,%rsp
    892e:	5b                   	pop    %rbx
    892f:	5d                   	pop    %rbp
    8930:	c3                   	retq   
    8931:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8938:	ba 20 00 00 00       	mov    $0x20,%edx
    893d:	be 02 00 00 00       	mov    $0x2,%esi
    8942:	bf 00 00 00 80       	mov    $0x80000000,%edi
    8947:	e8 c4 fc ff ff       	callq  8610 <galois_single_multiply>
    894c:	89 05 36 57 00 00    	mov    %eax,0x5736(%rip)        # e088 <prim32>
    8952:	eb a4                	jmp    88f8 <reed_sol_galois_w32_region_multby_2+0x18>
    8954:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    895b:	00 00 00 00 
    895f:	90                   	nop

0000000000008960 <reed_sol_galois_w08_region_multby_2>:
    8960:	f3 0f 1e fa          	endbr64 
    8964:	55                   	push   %rbp
    8965:	89 f5                	mov    %esi,%ebp
    8967:	53                   	push   %rbx
    8968:	48 89 fb             	mov    %rdi,%rbx
    896b:	48 83 ec 08          	sub    $0x8,%rsp
    896f:	83 3d 0e 57 00 00 ff 	cmpl   $0xffffffff,0x570e(%rip)        # e084 <prim08>
    8976:	74 58                	je     89d0 <reed_sol_galois_w08_region_multby_2+0x70>
    8978:	48 63 f5             	movslq %ebp,%rsi
    897b:	48 01 de             	add    %rbx,%rsi
    897e:	48 39 f3             	cmp    %rsi,%rbx
    8981:	73 41                	jae    89c4 <reed_sol_galois_w08_region_multby_2+0x64>
    8983:	44 8b 0d f6 56 00 00 	mov    0x56f6(%rip),%r9d        # e080 <mask08_1>
    898a:	44 8b 05 eb 56 00 00 	mov    0x56eb(%rip),%r8d        # e07c <mask08_2>
    8991:	8b 3d ed 56 00 00    	mov    0x56ed(%rip),%edi        # e084 <prim08>
    8997:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    899e:	00 00 
    89a0:	8b 13                	mov    (%rbx),%edx
    89a2:	48 83 c3 04          	add    $0x4,%rbx
    89a6:	89 d1                	mov    %edx,%ecx
    89a8:	44 21 c1             	and    %r8d,%ecx
    89ab:	8d 04 09             	lea    (%rcx,%rcx,1),%eax
    89ae:	c1 e9 07             	shr    $0x7,%ecx
    89b1:	29 c8                	sub    %ecx,%eax
    89b3:	01 d2                	add    %edx,%edx
    89b5:	21 f8                	and    %edi,%eax
    89b7:	44 21 ca             	and    %r9d,%edx
    89ba:	31 d0                	xor    %edx,%eax
    89bc:	89 43 fc             	mov    %eax,-0x4(%rbx)
    89bf:	48 39 de             	cmp    %rbx,%rsi
    89c2:	77 dc                	ja     89a0 <reed_sol_galois_w08_region_multby_2+0x40>
    89c4:	48 83 c4 08          	add    $0x8,%rsp
    89c8:	5b                   	pop    %rbx
    89c9:	5d                   	pop    %rbp
    89ca:	c3                   	retq   
    89cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    89d0:	ba 08 00 00 00       	mov    $0x8,%edx
    89d5:	be 02 00 00 00       	mov    $0x2,%esi
    89da:	bf 80 00 00 00       	mov    $0x80,%edi
    89df:	e8 2c fc ff ff       	callq  8610 <galois_single_multiply>
    89e4:	c7 05 96 56 00 00 00 	movl   $0x0,0x5696(%rip)        # e084 <prim08>
    89eb:	00 00 00 
    89ee:	85 c0                	test   %eax,%eax
    89f0:	74 13                	je     8a05 <reed_sol_galois_w08_region_multby_2+0xa5>
    89f2:	31 d2                	xor    %edx,%edx
    89f4:	0f 1f 40 00          	nopl   0x0(%rax)
    89f8:	09 c2                	or     %eax,%edx
    89fa:	c1 e0 08             	shl    $0x8,%eax
    89fd:	75 f9                	jne    89f8 <reed_sol_galois_w08_region_multby_2+0x98>
    89ff:	89 15 7f 56 00 00    	mov    %edx,0x567f(%rip)        # e084 <prim08>
    8a05:	c7 05 71 56 00 00 fe 	movl   $0xfefefefe,0x5671(%rip)        # e080 <mask08_1>
    8a0c:	fe fe fe 
    8a0f:	c7 05 63 56 00 00 80 	movl   $0x80808080,0x5663(%rip)        # e07c <mask08_2>
    8a16:	80 80 80 
    8a19:	e9 5a ff ff ff       	jmpq   8978 <reed_sol_galois_w08_region_multby_2+0x18>
    8a1e:	66 90                	xchg   %ax,%ax

0000000000008a20 <reed_sol_galois_w16_region_multby_2>:
    8a20:	f3 0f 1e fa          	endbr64 
    8a24:	55                   	push   %rbp
    8a25:	89 f5                	mov    %esi,%ebp
    8a27:	53                   	push   %rbx
    8a28:	48 89 fb             	mov    %rdi,%rbx
    8a2b:	48 83 ec 08          	sub    $0x8,%rsp
    8a2f:	83 3d 42 56 00 00 ff 	cmpl   $0xffffffff,0x5642(%rip)        # e078 <prim16>
    8a36:	74 58                	je     8a90 <reed_sol_galois_w16_region_multby_2+0x70>
    8a38:	48 63 f5             	movslq %ebp,%rsi
    8a3b:	48 01 de             	add    %rbx,%rsi
    8a3e:	48 39 f3             	cmp    %rsi,%rbx
    8a41:	73 41                	jae    8a84 <reed_sol_galois_w16_region_multby_2+0x64>
    8a43:	44 8b 0d 2a 56 00 00 	mov    0x562a(%rip),%r9d        # e074 <mask16_1>
    8a4a:	44 8b 05 1f 56 00 00 	mov    0x561f(%rip),%r8d        # e070 <mask16_2>
    8a51:	8b 3d 21 56 00 00    	mov    0x5621(%rip),%edi        # e078 <prim16>
    8a57:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    8a5e:	00 00 
    8a60:	8b 13                	mov    (%rbx),%edx
    8a62:	48 83 c3 04          	add    $0x4,%rbx
    8a66:	89 d1                	mov    %edx,%ecx
    8a68:	44 21 c1             	and    %r8d,%ecx
    8a6b:	8d 04 09             	lea    (%rcx,%rcx,1),%eax
    8a6e:	c1 e9 0f             	shr    $0xf,%ecx
    8a71:	29 c8                	sub    %ecx,%eax
    8a73:	01 d2                	add    %edx,%edx
    8a75:	21 f8                	and    %edi,%eax
    8a77:	44 21 ca             	and    %r9d,%edx
    8a7a:	31 d0                	xor    %edx,%eax
    8a7c:	89 43 fc             	mov    %eax,-0x4(%rbx)
    8a7f:	48 39 de             	cmp    %rbx,%rsi
    8a82:	77 dc                	ja     8a60 <reed_sol_galois_w16_region_multby_2+0x40>
    8a84:	48 83 c4 08          	add    $0x8,%rsp
    8a88:	5b                   	pop    %rbx
    8a89:	5d                   	pop    %rbp
    8a8a:	c3                   	retq   
    8a8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8a90:	ba 10 00 00 00       	mov    $0x10,%edx
    8a95:	be 02 00 00 00       	mov    $0x2,%esi
    8a9a:	bf 00 80 00 00       	mov    $0x8000,%edi
    8a9f:	e8 6c fb ff ff       	callq  8610 <galois_single_multiply>
    8aa4:	c7 05 ca 55 00 00 00 	movl   $0x0,0x55ca(%rip)        # e078 <prim16>
    8aab:	00 00 00 
    8aae:	85 c0                	test   %eax,%eax
    8ab0:	74 13                	je     8ac5 <reed_sol_galois_w16_region_multby_2+0xa5>
    8ab2:	31 d2                	xor    %edx,%edx
    8ab4:	0f 1f 40 00          	nopl   0x0(%rax)
    8ab8:	09 c2                	or     %eax,%edx
    8aba:	c1 e0 10             	shl    $0x10,%eax
    8abd:	75 f9                	jne    8ab8 <reed_sol_galois_w16_region_multby_2+0x98>
    8abf:	89 15 b3 55 00 00    	mov    %edx,0x55b3(%rip)        # e078 <prim16>
    8ac5:	c7 05 a5 55 00 00 fe 	movl   $0xfffefffe,0x55a5(%rip)        # e074 <mask16_1>
    8acc:	ff fe ff 
    8acf:	c7 05 97 55 00 00 00 	movl   $0x80008000,0x5597(%rip)        # e070 <mask16_2>
    8ad6:	80 00 80 
    8ad9:	e9 5a ff ff ff       	jmpq   8a38 <reed_sol_galois_w16_region_multby_2+0x18>
    8ade:	66 90                	xchg   %ax,%ax

0000000000008ae0 <reed_sol_r6_encode>:
    8ae0:	f3 0f 1e fa          	endbr64 
    8ae4:	41 57                	push   %r15
    8ae6:	49 63 c0             	movslq %r8d,%rax
    8ae9:	49 89 cf             	mov    %rcx,%r15
    8aec:	41 56                	push   %r14
    8aee:	41 89 fe             	mov    %edi,%r14d
    8af1:	41 55                	push   %r13
    8af3:	49 89 d5             	mov    %rdx,%r13
    8af6:	41 54                	push   %r12
    8af8:	41 89 f4             	mov    %esi,%r12d
    8afb:	55                   	push   %rbp
    8afc:	53                   	push   %rbx
    8afd:	48 89 c3             	mov    %rax,%rbx
    8b00:	48 83 ec 28          	sub    $0x28,%rsp
    8b04:	48 8b 32             	mov    (%rdx),%rsi
    8b07:	48 8b 39             	mov    (%rcx),%rdi
    8b0a:	48 89 c2             	mov    %rax,%rdx
    8b0d:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    8b12:	e8 a9 88 ff ff       	callq  13c0 <memcpy@plt>
    8b17:	49 63 c6             	movslq %r14d,%rax
    8b1a:	49 8d 44 c5 f8       	lea    -0x8(%r13,%rax,8),%rax
    8b1f:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    8b24:	41 8d 46 fe          	lea    -0x2(%r14),%eax
    8b28:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    8b2c:	41 83 fe 01          	cmp    $0x1,%r14d
    8b30:	0f 8e ca 00 00 00    	jle    8c00 <reed_sol_r6_encode+0x120>
    8b36:	89 c2                	mov    %eax,%edx
    8b38:	49 8d 6d 08          	lea    0x8(%r13),%rbp
    8b3c:	4d 8d 74 d5 10       	lea    0x10(%r13,%rdx,8),%r14
    8b41:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8b48:	49 8b 3f             	mov    (%r15),%rdi
    8b4b:	48 8b 75 00          	mov    0x0(%rbp),%rsi
    8b4f:	89 d9                	mov    %ebx,%ecx
    8b51:	48 89 fa             	mov    %rdi,%rdx
    8b54:	48 83 c5 08          	add    $0x8,%rbp
    8b58:	e8 73 f5 ff ff       	callq  80d0 <galois_region_xor>
    8b5d:	49 39 ee             	cmp    %rbp,%r14
    8b60:	75 e6                	jne    8b48 <reed_sol_r6_encode+0x68>
    8b62:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    8b67:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8b6b:	48 8b 30             	mov    (%rax),%rsi
    8b6e:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    8b73:	e8 48 88 ff ff       	callq  13c0 <memcpy@plt>
    8b78:	48 63 6c 24 0c       	movslq 0xc(%rsp),%rbp
    8b7d:	41 83 fc 10          	cmp    $0x10,%r12d
    8b81:	74 1d                	je     8ba0 <reed_sol_r6_encode+0xc0>
    8b83:	41 83 fc 20          	cmp    $0x20,%r12d
    8b87:	74 67                	je     8bf0 <reed_sol_r6_encode+0x110>
    8b89:	41 83 fc 08          	cmp    $0x8,%r12d
    8b8d:	74 51                	je     8be0 <reed_sol_r6_encode+0x100>
    8b8f:	48 83 c4 28          	add    $0x28,%rsp
    8b93:	5b                   	pop    %rbx
    8b94:	5d                   	pop    %rbp
    8b95:	41 5c                	pop    %r12
    8b97:	41 5d                	pop    %r13
    8b99:	41 5e                	pop    %r14
    8b9b:	31 c0                	xor    %eax,%eax
    8b9d:	41 5f                	pop    %r15
    8b9f:	c3                   	retq   
    8ba0:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8ba4:	89 de                	mov    %ebx,%esi
    8ba6:	e8 75 fe ff ff       	callq  8a20 <reed_sol_galois_w16_region_multby_2>
    8bab:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8baf:	49 8b 74 ed 00       	mov    0x0(%r13,%rbp,8),%rsi
    8bb4:	89 d9                	mov    %ebx,%ecx
    8bb6:	48 89 fa             	mov    %rdi,%rdx
    8bb9:	48 ff cd             	dec    %rbp
    8bbc:	e8 0f f5 ff ff       	callq  80d0 <galois_region_xor>
    8bc1:	85 ed                	test   %ebp,%ebp
    8bc3:	79 b8                	jns    8b7d <reed_sol_r6_encode+0x9d>
    8bc5:	48 83 c4 28          	add    $0x28,%rsp
    8bc9:	5b                   	pop    %rbx
    8bca:	5d                   	pop    %rbp
    8bcb:	41 5c                	pop    %r12
    8bcd:	41 5d                	pop    %r13
    8bcf:	41 5e                	pop    %r14
    8bd1:	b8 01 00 00 00       	mov    $0x1,%eax
    8bd6:	41 5f                	pop    %r15
    8bd8:	c3                   	retq   
    8bd9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8be0:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8be4:	89 de                	mov    %ebx,%esi
    8be6:	e8 75 fd ff ff       	callq  8960 <reed_sol_galois_w08_region_multby_2>
    8beb:	eb be                	jmp    8bab <reed_sol_r6_encode+0xcb>
    8bed:	0f 1f 00             	nopl   (%rax)
    8bf0:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8bf4:	89 de                	mov    %ebx,%esi
    8bf6:	e8 e5 fc ff ff       	callq  88e0 <reed_sol_galois_w32_region_multby_2>
    8bfb:	eb ae                	jmp    8bab <reed_sol_r6_encode+0xcb>
    8bfd:	0f 1f 00             	nopl   (%rax)
    8c00:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    8c05:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8c09:	48 8b 30             	mov    (%rax),%rsi
    8c0c:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    8c11:	e8 aa 87 ff ff       	callq  13c0 <memcpy@plt>
    8c16:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    8c1a:	85 c0                	test   %eax,%eax
    8c1c:	0f 89 56 ff ff ff    	jns    8b78 <reed_sol_r6_encode+0x98>
    8c22:	eb a1                	jmp    8bc5 <reed_sol_r6_encode+0xe5>
    8c24:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    8c2b:	00 00 00 00 
    8c2f:	90                   	nop

0000000000008c30 <reed_sol_extended_vandermonde_matrix>:
    8c30:	f3 0f 1e fa          	endbr64 
    8c34:	41 57                	push   %r15
    8c36:	41 56                	push   %r14
    8c38:	41 55                	push   %r13
    8c3a:	41 54                	push   %r12
    8c3c:	41 89 f4             	mov    %esi,%r12d
    8c3f:	55                   	push   %rbp
    8c40:	89 fd                	mov    %edi,%ebp
    8c42:	53                   	push   %rbx
    8c43:	89 d3                	mov    %edx,%ebx
    8c45:	48 83 ec 18          	sub    $0x18,%rsp
    8c49:	83 fa 1d             	cmp    $0x1d,%edx
    8c4c:	7f 1a                	jg     8c68 <reed_sol_extended_vandermonde_matrix+0x38>
    8c4e:	b8 01 00 00 00       	mov    $0x1,%eax
    8c53:	c4 e2 69 f7 c0       	shlx   %edx,%eax,%eax
    8c58:	39 f8                	cmp    %edi,%eax
    8c5a:	0f 8c 19 01 00 00    	jl     8d79 <reed_sol_extended_vandermonde_matrix+0x149>
    8c60:	39 f0                	cmp    %esi,%eax
    8c62:	0f 8c 11 01 00 00    	jl     8d79 <reed_sol_extended_vandermonde_matrix+0x149>
    8c68:	41 89 ee             	mov    %ebp,%r14d
    8c6b:	45 0f af f4          	imul   %r12d,%r14d
    8c6f:	49 63 fe             	movslq %r14d,%rdi
    8c72:	48 c1 e7 02          	shl    $0x2,%rdi
    8c76:	e8 85 87 ff ff       	callq  1400 <malloc@plt>
    8c7b:	49 89 c5             	mov    %rax,%r13
    8c7e:	48 85 c0             	test   %rax,%rax
    8c81:	0f 84 f2 00 00 00    	je     8d79 <reed_sol_extended_vandermonde_matrix+0x149>
    8c87:	c7 00 01 00 00 00    	movl   $0x1,(%rax)
    8c8d:	41 83 fc 01          	cmp    $0x1,%r12d
    8c91:	0f 8e e7 00 00 00    	jle    8d7e <reed_sol_extended_vandermonde_matrix+0x14e>
    8c97:	41 8d 54 24 fe       	lea    -0x2(%r12),%edx
    8c9c:	48 8d 40 04          	lea    0x4(%rax),%rax
    8ca0:	49 8d 54 95 08       	lea    0x8(%r13,%rdx,4),%rdx
    8ca5:	0f 1f 00             	nopl   (%rax)
    8ca8:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
    8cae:	48 83 c0 04          	add    $0x4,%rax
    8cb2:	48 39 d0             	cmp    %rdx,%rax
    8cb5:	75 f1                	jne    8ca8 <reed_sol_extended_vandermonde_matrix+0x78>
    8cb7:	83 fd 01             	cmp    $0x1,%ebp
    8cba:	0f 84 a7 00 00 00    	je     8d67 <reed_sol_extended_vandermonde_matrix+0x137>
    8cc0:	8d 45 ff             	lea    -0x1(%rbp),%eax
    8cc3:	89 44 24 08          	mov    %eax,0x8(%rsp)
    8cc7:	44 89 f0             	mov    %r14d,%eax
    8cca:	44 29 e0             	sub    %r12d,%eax
    8ccd:	48 98                	cltq   
    8ccf:	49 8d 54 85 00       	lea    0x0(%r13,%rax,4),%rdx
    8cd4:	41 8d 4c 24 ff       	lea    -0x1(%r12),%ecx
    8cd9:	31 c0                	xor    %eax,%eax
    8cdb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8ce0:	c7 04 82 00 00 00 00 	movl   $0x0,(%rdx,%rax,4)
    8ce7:	48 ff c0             	inc    %rax
    8cea:	39 c1                	cmp    %eax,%ecx
    8cec:	7f f2                	jg     8ce0 <reed_sol_extended_vandermonde_matrix+0xb0>
    8cee:	41 ff ce             	dec    %r14d
    8cf1:	4d 63 f6             	movslq %r14d,%r14
    8cf4:	43 c7 44 b5 00 01 00 	movl   $0x1,0x0(%r13,%r14,4)
    8cfb:	00 00 
    8cfd:	83 fd 02             	cmp    $0x2,%ebp
    8d00:	74 65                	je     8d67 <reed_sol_extended_vandermonde_matrix+0x137>
    8d02:	83 7c 24 08 01       	cmpl   $0x1,0x8(%rsp)
    8d07:	7e 5e                	jle    8d67 <reed_sol_extended_vandermonde_matrix+0x137>
    8d09:	41 8d 44 24 ff       	lea    -0x1(%r12),%eax
    8d0e:	44 89 64 24 04       	mov    %r12d,0x4(%rsp)
    8d13:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    8d17:	41 bf 01 00 00 00    	mov    $0x1,%r15d
    8d1d:	0f 1f 00             	nopl   (%rax)
    8d20:	45 85 e4             	test   %r12d,%r12d
    8d23:	7e 33                	jle    8d58 <reed_sol_extended_vandermonde_matrix+0x128>
    8d25:	48 63 54 24 04       	movslq 0x4(%rsp),%rdx
    8d2a:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    8d2e:	4d 8d 74 95 00       	lea    0x0(%r13,%rdx,4),%r14
    8d33:	48 01 d0             	add    %rdx,%rax
    8d36:	49 8d 6c 85 04       	lea    0x4(%r13,%rax,4),%rbp
    8d3b:	bf 01 00 00 00       	mov    $0x1,%edi
    8d40:	41 89 3e             	mov    %edi,(%r14)
    8d43:	89 da                	mov    %ebx,%edx
    8d45:	44 89 fe             	mov    %r15d,%esi
    8d48:	e8 c3 f8 ff ff       	callq  8610 <galois_single_multiply>
    8d4d:	49 83 c6 04          	add    $0x4,%r14
    8d51:	89 c7                	mov    %eax,%edi
    8d53:	4c 39 f5             	cmp    %r14,%rbp
    8d56:	75 e8                	jne    8d40 <reed_sol_extended_vandermonde_matrix+0x110>
    8d58:	41 ff c7             	inc    %r15d
    8d5b:	44 01 64 24 04       	add    %r12d,0x4(%rsp)
    8d60:	44 39 7c 24 08       	cmp    %r15d,0x8(%rsp)
    8d65:	75 b9                	jne    8d20 <reed_sol_extended_vandermonde_matrix+0xf0>
    8d67:	48 83 c4 18          	add    $0x18,%rsp
    8d6b:	5b                   	pop    %rbx
    8d6c:	5d                   	pop    %rbp
    8d6d:	41 5c                	pop    %r12
    8d6f:	4c 89 e8             	mov    %r13,%rax
    8d72:	41 5d                	pop    %r13
    8d74:	41 5e                	pop    %r14
    8d76:	41 5f                	pop    %r15
    8d78:	c3                   	retq   
    8d79:	45 31 ed             	xor    %r13d,%r13d
    8d7c:	eb e9                	jmp    8d67 <reed_sol_extended_vandermonde_matrix+0x137>
    8d7e:	83 fd 01             	cmp    $0x1,%ebp
    8d81:	74 e4                	je     8d67 <reed_sol_extended_vandermonde_matrix+0x137>
    8d83:	8d 45 ff             	lea    -0x1(%rbp),%eax
    8d86:	89 44 24 08          	mov    %eax,0x8(%rsp)
    8d8a:	45 29 e6             	sub    %r12d,%r14d
    8d8d:	e9 5f ff ff ff       	jmpq   8cf1 <reed_sol_extended_vandermonde_matrix+0xc1>
    8d92:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    8d99:	00 00 00 00 
    8d9d:	0f 1f 00             	nopl   (%rax)

0000000000008da0 <reed_sol_big_vandermonde_distribution_matrix>:
    8da0:	f3 0f 1e fa          	endbr64 
    8da4:	41 57                	push   %r15
    8da6:	41 56                	push   %r14
    8da8:	41 55                	push   %r13
    8daa:	41 54                	push   %r12
    8dac:	55                   	push   %rbp
    8dad:	53                   	push   %rbx
    8dae:	48 83 ec 78          	sub    $0x78,%rsp
    8db2:	89 7c 24 04          	mov    %edi,0x4(%rsp)
    8db6:	89 74 24 6c          	mov    %esi,0x6c(%rsp)
    8dba:	39 fe                	cmp    %edi,%esi
    8dbc:	0f 8d ed 03 00 00    	jge    91af <reed_sol_big_vandermonde_distribution_matrix+0x40f>
    8dc2:	89 f3                	mov    %esi,%ebx
    8dc4:	41 89 d6             	mov    %edx,%r14d
    8dc7:	e8 64 fe ff ff       	callq  8c30 <reed_sol_extended_vandermonde_matrix>
    8dcc:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    8dd1:	48 89 c1             	mov    %rax,%rcx
    8dd4:	48 85 c0             	test   %rax,%rax
    8dd7:	0f 84 d2 03 00 00    	je     91af <reed_sol_big_vandermonde_distribution_matrix+0x40f>
    8ddd:	83 fb 01             	cmp    $0x1,%ebx
    8de0:	0f 8e ca 01 00 00    	jle    8fb0 <reed_sol_big_vandermonde_distribution_matrix+0x210>
    8de6:	48 63 fb             	movslq %ebx,%rdi
    8de9:	8d 43 fe             	lea    -0x2(%rbx),%eax
    8dec:	48 8d 14 bd 00 00 00 	lea    0x0(,%rdi,4),%rdx
    8df3:	00 
    8df4:	48 83 c0 02          	add    $0x2,%rax
    8df8:	48 01 d1             	add    %rdx,%rcx
    8dfb:	48 89 54 24 08       	mov    %rdx,0x8(%rsp)
    8e00:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    8e05:	48 8d 57 01          	lea    0x1(%rdi),%rdx
    8e09:	8d 43 ff             	lea    -0x1(%rbx),%eax
    8e0c:	48 89 7c 24 48       	mov    %rdi,0x48(%rsp)
    8e11:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    8e16:	48 89 54 24 58       	mov    %rdx,0x58(%rsp)
    8e1b:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
    8e20:	48 89 7c 24 50       	mov    %rdi,0x50(%rsp)
    8e25:	48 c7 44 24 20 01 00 	movq   $0x1,0x20(%rsp)
    8e2c:	00 00 
    8e2e:	89 44 24 68          	mov    %eax,0x68(%rsp)
    8e32:	44 89 74 24 44       	mov    %r14d,0x44(%rsp)
    8e37:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    8e3e:	00 00 
    8e40:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    8e45:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    8e4a:	89 fe                	mov    %edi,%esi
    8e4c:	89 fa                	mov    %edi,%edx
    8e4e:	4c 8b 44 24 48       	mov    0x48(%rsp),%r8
    8e53:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    8e58:	44 8b 4c 24 04       	mov    0x4(%rsp),%r9d
    8e5d:	0f 1f 00             	nopl   (%rax)
    8e60:	44 8b 14 87          	mov    (%rdi,%rax,4),%r10d
    8e64:	89 c1                	mov    %eax,%ecx
    8e66:	45 85 d2             	test   %r10d,%r10d
    8e69:	75 45                	jne    8eb0 <reed_sol_big_vandermonde_distribution_matrix+0x110>
    8e6b:	ff c2                	inc    %edx
    8e6d:	4c 01 c0             	add    %r8,%rax
    8e70:	41 39 d1             	cmp    %edx,%r9d
    8e73:	7f eb                	jg     8e60 <reed_sol_big_vandermonde_distribution_matrix+0xc0>
    8e75:	44 8b 74 24 44       	mov    0x44(%rsp),%r14d
    8e7a:	48 8b 3d bf 82 00 00 	mov    0x82bf(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    8e81:	44 8b 44 24 6c       	mov    0x6c(%rsp),%r8d
    8e86:	8b 4c 24 04          	mov    0x4(%rsp),%ecx
    8e8a:	45 89 f1             	mov    %r14d,%r9d
    8e8d:	48 8d 15 d4 1e 00 00 	lea    0x1ed4(%rip),%rdx        # ad68 <prim_poly+0x88>
    8e94:	be 01 00 00 00       	mov    $0x1,%esi
    8e99:	31 c0                	xor    %eax,%eax
    8e9b:	e8 e0 85 ff ff       	callq  1480 <__fprintf_chk@plt>
    8ea0:	bf 01 00 00 00       	mov    $0x1,%edi
    8ea5:	e8 b6 85 ff ff       	callq  1460 <exit@plt>
    8eaa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    8eb0:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    8eb5:	89 44 24 40          	mov    %eax,0x40(%rsp)
    8eb9:	39 c2                	cmp    %eax,%edx
    8ebb:	0f 85 4f 02 00 00    	jne    9110 <reed_sol_big_vandermonde_distribution_matrix+0x370>
    8ec1:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    8ec6:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    8ecb:	8b 34 88             	mov    (%rax,%rcx,4),%esi
    8ece:	83 fe 01             	cmp    $0x1,%esi
    8ed1:	0f 85 79 02 00 00    	jne    9150 <reed_sol_big_vandermonde_distribution_matrix+0x3b0>
    8ed7:	8b 44 24 68          	mov    0x68(%rsp),%eax
    8edb:	45 31 ff             	xor    %r15d,%r15d
    8ede:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    8ee3:	eb 15                	jmp    8efa <reed_sol_big_vandermonde_distribution_matrix+0x15a>
    8ee5:	0f 1f 00             	nopl   (%rax)
    8ee8:	49 8d 47 01          	lea    0x1(%r15),%rax
    8eec:	4c 3b 7c 24 18       	cmp    0x18(%rsp),%r15
    8ef1:	0f 84 81 00 00 00    	je     8f78 <reed_sol_big_vandermonde_distribution_matrix+0x1d8>
    8ef7:	49 89 c7             	mov    %rax,%r15
    8efa:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    8eff:	46 8b 2c b8          	mov    (%rax,%r15,4),%r13d
    8f03:	44 39 7c 24 40       	cmp    %r15d,0x40(%rsp)
    8f08:	74 de                	je     8ee8 <reed_sol_big_vandermonde_distribution_matrix+0x148>
    8f0a:	45 85 ed             	test   %r13d,%r13d
    8f0d:	74 d9                	je     8ee8 <reed_sol_big_vandermonde_distribution_matrix+0x148>
    8f0f:	8b 54 24 04          	mov    0x4(%rsp),%edx
    8f13:	85 d2                	test   %edx,%edx
    8f15:	7e d1                	jle    8ee8 <reed_sol_big_vandermonde_distribution_matrix+0x148>
    8f17:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    8f1c:	4c 8b 74 24 20       	mov    0x20(%rsp),%r14
    8f21:	4a 8d 2c b8          	lea    (%rax,%r15,4),%rbp
    8f25:	4d 29 fe             	sub    %r15,%r14
    8f28:	45 31 e4             	xor    %r12d,%r12d
    8f2b:	4c 89 7c 24 30       	mov    %r15,0x30(%rsp)
    8f30:	49 89 ef             	mov    %rbp,%r15
    8f33:	44 89 e5             	mov    %r12d,%ebp
    8f36:	4d 89 f4             	mov    %r14,%r12
    8f39:	44 8b 74 24 44       	mov    0x44(%rsp),%r14d
    8f3e:	66 90                	xchg   %ax,%ax
    8f40:	43 8b 34 a7          	mov    (%r15,%r12,4),%esi
    8f44:	41 8b 1f             	mov    (%r15),%ebx
    8f47:	44 89 f2             	mov    %r14d,%edx
    8f4a:	44 89 ef             	mov    %r13d,%edi
    8f4d:	e8 be f6 ff ff       	callq  8610 <galois_single_multiply>
    8f52:	31 c3                	xor    %eax,%ebx
    8f54:	ff c5                	inc    %ebp
    8f56:	41 89 1f             	mov    %ebx,(%r15)
    8f59:	4c 03 7c 24 08       	add    0x8(%rsp),%r15
    8f5e:	39 6c 24 04          	cmp    %ebp,0x4(%rsp)
    8f62:	75 dc                	jne    8f40 <reed_sol_big_vandermonde_distribution_matrix+0x1a0>
    8f64:	4c 8b 7c 24 30       	mov    0x30(%rsp),%r15
    8f69:	49 8d 47 01          	lea    0x1(%r15),%rax
    8f6d:	4c 3b 7c 24 18       	cmp    0x18(%rsp),%r15
    8f72:	0f 85 7f ff ff ff    	jne    8ef7 <reed_sol_big_vandermonde_distribution_matrix+0x157>
    8f78:	48 ff 44 24 20       	incq   0x20(%rsp)
    8f7d:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    8f82:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    8f87:	48 01 7c 24 10       	add    %rdi,0x10(%rsp)
    8f8c:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    8f91:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    8f96:	48 01 4c 24 50       	add    %rcx,0x50(%rsp)
    8f9b:	48 01 7c 24 38       	add    %rdi,0x38(%rsp)
    8fa0:	48 3b 44 24 60       	cmp    0x60(%rsp),%rax
    8fa5:	0f 85 95 fe ff ff    	jne    8e40 <reed_sol_big_vandermonde_distribution_matrix+0xa0>
    8fab:	44 8b 74 24 44       	mov    0x44(%rsp),%r14d
    8fb0:	8b 4c 24 6c          	mov    0x6c(%rsp),%ecx
    8fb4:	89 c8                	mov    %ecx,%eax
    8fb6:	0f af c1             	imul   %ecx,%eax
    8fb9:	85 c9                	test   %ecx,%ecx
    8fbb:	0f 8e 9d 00 00 00    	jle    905e <reed_sol_big_vandermonde_distribution_matrix+0x2be>
    8fc1:	48 63 d8             	movslq %eax,%rbx
    8fc4:	48 63 c1             	movslq %ecx,%rax
    8fc7:	44 8d 68 ff          	lea    -0x1(%rax),%r13d
    8fcb:	48 8d 6b 01          	lea    0x1(%rbx),%rbp
    8fcf:	4d 8d 64 2d 00       	lea    0x0(%r13,%rbp,1),%r12
    8fd4:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    8fd9:	4c 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%r15
    8fe0:	00 
    8fe1:	48 89 d9             	mov    %rbx,%rcx
    8fe4:	eb 17                	jmp    8ffd <reed_sol_big_vandermonde_distribution_matrix+0x25d>
    8fe6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    8fed:	00 00 00 
    8ff0:	48 89 e9             	mov    %rbp,%rcx
    8ff3:	48 39 6c 24 08       	cmp    %rbp,0x8(%rsp)
    8ff8:	74 64                	je     905e <reed_sol_big_vandermonde_distribution_matrix+0x2be>
    8ffa:	48 ff c5             	inc    %rbp
    8ffd:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9002:	8b 34 88             	mov    (%rax,%rcx,4),%esi
    9005:	83 fe 01             	cmp    $0x1,%esi
    9008:	74 e6                	je     8ff0 <reed_sol_big_vandermonde_distribution_matrix+0x250>
    900a:	44 89 f2             	mov    %r14d,%edx
    900d:	bf 01 00 00 00       	mov    $0x1,%edi
    9012:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    9017:	e8 14 f6 ff ff       	callq  8630 <galois_single_divide>
    901c:	41 89 c5             	mov    %eax,%r13d
    901f:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    9024:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9029:	44 8b 64 24 6c       	mov    0x6c(%rsp),%r12d
    902e:	48 8d 1c 88          	lea    (%rax,%rcx,4),%rbx
    9032:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    9038:	8b 33                	mov    (%rbx),%esi
    903a:	44 89 f2             	mov    %r14d,%edx
    903d:	44 89 ef             	mov    %r13d,%edi
    9040:	e8 cb f5 ff ff       	callq  8610 <galois_single_multiply>
    9045:	41 ff c4             	inc    %r12d
    9048:	89 03                	mov    %eax,(%rbx)
    904a:	4c 01 fb             	add    %r15,%rbx
    904d:	44 39 64 24 04       	cmp    %r12d,0x4(%rsp)
    9052:	75 e4                	jne    9038 <reed_sol_big_vandermonde_distribution_matrix+0x298>
    9054:	48 89 e9             	mov    %rbp,%rcx
    9057:	48 39 6c 24 08       	cmp    %rbp,0x8(%rsp)
    905c:	75 9c                	jne    8ffa <reed_sol_big_vandermonde_distribution_matrix+0x25a>
    905e:	8b 44 24 6c          	mov    0x6c(%rsp),%eax
    9062:	8d 48 01             	lea    0x1(%rax),%ecx
    9065:	89 cd                	mov    %ecx,%ebp
    9067:	0f af e8             	imul   %eax,%ebp
    906a:	3b 4c 24 04          	cmp    0x4(%rsp),%ecx
    906e:	0f 8d 44 01 00 00    	jge    91b8 <reed_sol_big_vandermonde_distribution_matrix+0x418>
    9074:	4c 63 f8             	movslq %eax,%r15
    9077:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    907c:	48 63 ed             	movslq %ebp,%rbp
    907f:	ff c8                	dec    %eax
    9081:	4e 8d 24 bd 00 00 00 	lea    0x0(,%r15,4),%r12
    9088:	00 
    9089:	48 8d 44 05 01       	lea    0x1(%rbp,%rax,1),%rax
    908e:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    9093:	4c 89 7c 24 10       	mov    %r15,0x10(%rsp)
    9098:	48 8d 1c 87          	lea    (%rdi,%rax,4),%rbx
    909c:	41 89 cc             	mov    %ecx,%r12d
    909f:	eb 1f                	jmp    90c0 <reed_sol_big_vandermonde_distribution_matrix+0x320>
    90a1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    90a8:	41 ff c4             	inc    %r12d
    90ab:	48 03 6c 24 10       	add    0x10(%rsp),%rbp
    90b0:	48 03 5c 24 08       	add    0x8(%rsp),%rbx
    90b5:	44 39 64 24 04       	cmp    %r12d,0x4(%rsp)
    90ba:	0f 84 f8 00 00 00    	je     91b8 <reed_sol_big_vandermonde_distribution_matrix+0x418>
    90c0:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    90c5:	8b 34 a8             	mov    (%rax,%rbp,4),%esi
    90c8:	83 fe 01             	cmp    $0x1,%esi
    90cb:	74 db                	je     90a8 <reed_sol_big_vandermonde_distribution_matrix+0x308>
    90cd:	44 89 f2             	mov    %r14d,%edx
    90d0:	bf 01 00 00 00       	mov    $0x1,%edi
    90d5:	e8 56 f5 ff ff       	callq  8630 <galois_single_divide>
    90da:	41 89 c5             	mov    %eax,%r13d
    90dd:	8b 44 24 6c          	mov    0x6c(%rsp),%eax
    90e1:	85 c0                	test   %eax,%eax
    90e3:	7e c3                	jle    90a8 <reed_sol_big_vandermonde_distribution_matrix+0x308>
    90e5:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    90ea:	4c 8d 3c a8          	lea    (%rax,%rbp,4),%r15
    90ee:	66 90                	xchg   %ax,%ax
    90f0:	41 8b 3f             	mov    (%r15),%edi
    90f3:	44 89 f2             	mov    %r14d,%edx
    90f6:	44 89 ee             	mov    %r13d,%esi
    90f9:	e8 12 f5 ff ff       	callq  8610 <galois_single_multiply>
    90fe:	41 89 07             	mov    %eax,(%r15)
    9101:	49 83 c7 04          	add    $0x4,%r15
    9105:	49 39 df             	cmp    %rbx,%r15
    9108:	75 e6                	jne    90f0 <reed_sol_big_vandermonde_distribution_matrix+0x350>
    910a:	eb 9c                	jmp    90a8 <reed_sol_big_vandermonde_distribution_matrix+0x308>
    910c:	0f 1f 40 00          	nopl   0x0(%rax)
    9110:	8b 54 24 68          	mov    0x68(%rsp),%edx
    9114:	29 f1                	sub    %esi,%ecx
    9116:	48 63 c9             	movslq %ecx,%rcx
    9119:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    911e:	48 01 ca             	add    %rcx,%rdx
    9121:	48 8d 04 8f          	lea    (%rdi,%rcx,4),%rax
    9125:	48 8d 7c 97 04       	lea    0x4(%rdi,%rdx,4),%rdi
    912a:	48 8b 54 24 50       	mov    0x50(%rsp),%rdx
    912f:	48 29 ca             	sub    %rcx,%rdx
    9132:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    9138:	8b 08                	mov    (%rax),%ecx
    913a:	8b 34 90             	mov    (%rax,%rdx,4),%esi
    913d:	89 30                	mov    %esi,(%rax)
    913f:	89 0c 90             	mov    %ecx,(%rax,%rdx,4)
    9142:	48 83 c0 04          	add    $0x4,%rax
    9146:	48 39 f8             	cmp    %rdi,%rax
    9149:	75 ed                	jne    9138 <reed_sol_big_vandermonde_distribution_matrix+0x398>
    914b:	e9 71 fd ff ff       	jmpq   8ec1 <reed_sol_big_vandermonde_distribution_matrix+0x121>
    9150:	44 8b 74 24 44       	mov    0x44(%rsp),%r14d
    9155:	bf 01 00 00 00       	mov    $0x1,%edi
    915a:	44 89 f2             	mov    %r14d,%edx
    915d:	e8 ce f4 ff ff       	callq  8630 <galois_single_divide>
    9162:	44 8b 6c 24 04       	mov    0x4(%rsp),%r13d
    9167:	89 c5                	mov    %eax,%ebp
    9169:	45 85 ed             	test   %r13d,%r13d
    916c:	0f 8e 65 fd ff ff    	jle    8ed7 <reed_sol_big_vandermonde_distribution_matrix+0x137>
    9172:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9177:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    917c:	4c 8b 64 24 08       	mov    0x8(%rsp),%r12
    9181:	4c 8d 3c 88          	lea    (%rax,%rcx,4),%r15
    9185:	31 db                	xor    %ebx,%ebx
    9187:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    918e:	00 00 
    9190:	41 8b 37             	mov    (%r15),%esi
    9193:	44 89 f2             	mov    %r14d,%edx
    9196:	89 ef                	mov    %ebp,%edi
    9198:	e8 73 f4 ff ff       	callq  8610 <galois_single_multiply>
    919d:	ff c3                	inc    %ebx
    919f:	41 89 07             	mov    %eax,(%r15)
    91a2:	4d 01 e7             	add    %r12,%r15
    91a5:	41 39 dd             	cmp    %ebx,%r13d
    91a8:	75 e6                	jne    9190 <reed_sol_big_vandermonde_distribution_matrix+0x3f0>
    91aa:	e9 28 fd ff ff       	jmpq   8ed7 <reed_sol_big_vandermonde_distribution_matrix+0x137>
    91af:	48 c7 44 24 28 00 00 	movq   $0x0,0x28(%rsp)
    91b6:	00 00 
    91b8:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    91bd:	48 83 c4 78          	add    $0x78,%rsp
    91c1:	5b                   	pop    %rbx
    91c2:	5d                   	pop    %rbp
    91c3:	41 5c                	pop    %r12
    91c5:	41 5d                	pop    %r13
    91c7:	41 5e                	pop    %r14
    91c9:	41 5f                	pop    %r15
    91cb:	c3                   	retq   
    91cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000091d0 <reed_sol_vandermonde_coding_matrix>:
    91d0:	f3 0f 1e fa          	endbr64 
    91d4:	41 55                	push   %r13
    91d6:	41 54                	push   %r12
    91d8:	55                   	push   %rbp
    91d9:	89 f5                	mov    %esi,%ebp
    91db:	53                   	push   %rbx
    91dc:	89 fb                	mov    %edi,%ebx
    91de:	01 f7                	add    %esi,%edi
    91e0:	48 83 ec 08          	sub    $0x8,%rsp
    91e4:	89 de                	mov    %ebx,%esi
    91e6:	e8 b5 fb ff ff       	callq  8da0 <reed_sol_big_vandermonde_distribution_matrix>
    91eb:	48 85 c0             	test   %rax,%rax
    91ee:	74 60                	je     9250 <reed_sol_vandermonde_coding_matrix+0x80>
    91f0:	0f af eb             	imul   %ebx,%ebp
    91f3:	49 89 c5             	mov    %rax,%r13
    91f6:	48 63 fd             	movslq %ebp,%rdi
    91f9:	48 c1 e7 02          	shl    $0x2,%rdi
    91fd:	e8 fe 81 ff ff       	callq  1400 <malloc@plt>
    9202:	49 89 c4             	mov    %rax,%r12
    9205:	48 85 c0             	test   %rax,%rax
    9208:	74 28                	je     9232 <reed_sol_vandermonde_coding_matrix+0x62>
    920a:	0f af db             	imul   %ebx,%ebx
    920d:	85 ed                	test   %ebp,%ebp
    920f:	7e 21                	jle    9232 <reed_sol_vandermonde_coding_matrix+0x62>
    9211:	48 63 db             	movslq %ebx,%rbx
    9214:	8d 75 ff             	lea    -0x1(%rbp),%esi
    9217:	49 8d 44 9d 00       	lea    0x0(%r13,%rbx,4),%rax
    921c:	31 d2                	xor    %edx,%edx
    921e:	66 90                	xchg   %ax,%ax
    9220:	8b 0c 90             	mov    (%rax,%rdx,4),%ecx
    9223:	41 89 0c 94          	mov    %ecx,(%r12,%rdx,4)
    9227:	48 89 d1             	mov    %rdx,%rcx
    922a:	48 ff c2             	inc    %rdx
    922d:	48 39 ce             	cmp    %rcx,%rsi
    9230:	75 ee                	jne    9220 <reed_sol_vandermonde_coding_matrix+0x50>
    9232:	4c 89 ef             	mov    %r13,%rdi
    9235:	e8 46 80 ff ff       	callq  1280 <free@plt>
    923a:	48 83 c4 08          	add    $0x8,%rsp
    923e:	5b                   	pop    %rbx
    923f:	5d                   	pop    %rbp
    9240:	4c 89 e0             	mov    %r12,%rax
    9243:	41 5c                	pop    %r12
    9245:	41 5d                	pop    %r13
    9247:	c3                   	retq   
    9248:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    924f:	00 
    9250:	48 83 c4 08          	add    $0x8,%rsp
    9254:	5b                   	pop    %rbx
    9255:	45 31 e4             	xor    %r12d,%r12d
    9258:	5d                   	pop    %rbp
    9259:	4c 89 e0             	mov    %r12,%rax
    925c:	41 5c                	pop    %r12
    925e:	41 5d                	pop    %r13
    9260:	c3                   	retq   
    9261:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    9268:	00 00 00 
    926b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000009270 <cauchy_n_ones>:
    9270:	f3 0f 1e fa          	endbr64 
    9274:	41 57                	push   %r15
    9276:	41 56                	push   %r14
    9278:	4c 8d 35 21 7e 00 00 	lea    0x7e21(%rip),%r14        # 110a0 <PPs>
    927f:	41 55                	push   %r13
    9281:	4c 63 ee             	movslq %esi,%r13
    9284:	41 8d 45 ff          	lea    -0x1(%r13),%eax
    9288:	41 54                	push   %r12
    928a:	41 bc 01 00 00 00    	mov    $0x1,%r12d
    9290:	c4 42 79 f7 e4       	shlx   %eax,%r12d,%r12d
    9295:	55                   	push   %rbp
    9296:	4c 89 ed             	mov    %r13,%rbp
    9299:	53                   	push   %rbx
    929a:	89 fb                	mov    %edi,%ebx
    929c:	48 83 ec 08          	sub    $0x8,%rsp
    92a0:	43 83 3c ae ff       	cmpl   $0xffffffff,(%r14,%r13,4)
    92a5:	0f 84 cd 00 00 00    	je     9378 <cauchy_n_ones+0x108>
    92ab:	85 ed                	test   %ebp,%ebp
    92ad:	0f 8e 2c 01 00 00    	jle    93df <cauchy_n_ones+0x16f>
    92b3:	31 c0                	xor    %eax,%eax
    92b5:	31 c9                	xor    %ecx,%ecx
    92b7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    92be:	00 00 
    92c0:	c4 e2 7a f7 d3       	sarx   %eax,%ebx,%edx
    92c5:	83 e2 01             	and    $0x1,%edx
    92c8:	83 fa 01             	cmp    $0x1,%edx
    92cb:	83 d9 ff             	sbb    $0xffffffff,%ecx
    92ce:	ff c0                	inc    %eax
    92d0:	39 c5                	cmp    %eax,%ebp
    92d2:	75 ec                	jne    92c0 <cauchy_n_ones+0x50>
    92d4:	83 fd 01             	cmp    $0x1,%ebp
    92d7:	0f 8e 0e 01 00 00    	jle    93eb <cauchy_n_ones+0x17b>
    92dd:	4d 69 d5 84 00 00 00 	imul   $0x84,%r13,%r10
    92e4:	4d 89 e9             	mov    %r13,%r9
    92e7:	48 8d 05 72 84 00 00 	lea    0x8472(%rip),%rax        # 11760 <ONEs>
    92ee:	49 c1 e1 05          	shl    $0x5,%r9
    92f2:	49 01 c2             	add    %rax,%r10
    92f5:	4d 01 e9             	add    %r13,%r9
    92f8:	41 89 c8             	mov    %ecx,%r8d
    92fb:	be 01 00 00 00       	mov    $0x1,%esi
    9300:	4c 8d 1d 79 95 00 00 	lea    0x9579(%rip),%r11        # 12880 <NOs>
    9307:	4c 8d 78 04          	lea    0x4(%rax),%r15
    930b:	eb 0e                	jmp    931b <cauchy_n_ones+0xab>
    930d:	0f 1f 00             	nopl   (%rax)
    9310:	01 db                	add    %ebx,%ebx
    9312:	ff c6                	inc    %esi
    9314:	41 01 c8             	add    %ecx,%r8d
    9317:	39 f5                	cmp    %esi,%ebp
    9319:	74 45                	je     9360 <cauchy_n_ones+0xf0>
    931b:	41 85 dc             	test   %ebx,%r12d
    931e:	74 f0                	je     9310 <cauchy_n_ones+0xa0>
    9320:	44 31 e3             	xor    %r12d,%ebx
    9323:	43 8b 04 ab          	mov    (%r11,%r13,4),%eax
    9327:	01 db                	add    %ebx,%ebx
    9329:	43 33 1c ae          	xor    (%r14,%r13,4),%ebx
    932d:	ff c9                	dec    %ecx
    932f:	85 c0                	test   %eax,%eax
    9331:	7e df                	jle    9312 <cauchy_n_ones+0xa2>
    9333:	ff c8                	dec    %eax
    9335:	4c 01 c8             	add    %r9,%rax
    9338:	49 8d 3c 87          	lea    (%r15,%rax,4),%rdi
    933c:	4c 89 d0             	mov    %r10,%rax
    933f:	90                   	nop
    9340:	8b 10                	mov    (%rax),%edx
    9342:	21 da                	and    %ebx,%edx
    9344:	83 fa 01             	cmp    $0x1,%edx
    9347:	19 d2                	sbb    %edx,%edx
    9349:	83 ca 01             	or     $0x1,%edx
    934c:	48 83 c0 04          	add    $0x4,%rax
    9350:	01 d1                	add    %edx,%ecx
    9352:	48 39 f8             	cmp    %rdi,%rax
    9355:	75 e9                	jne    9340 <cauchy_n_ones+0xd0>
    9357:	ff c6                	inc    %esi
    9359:	41 01 c8             	add    %ecx,%r8d
    935c:	39 f5                	cmp    %esi,%ebp
    935e:	75 bb                	jne    931b <cauchy_n_ones+0xab>
    9360:	48 83 c4 08          	add    $0x8,%rsp
    9364:	5b                   	pop    %rbx
    9365:	5d                   	pop    %rbp
    9366:	41 5c                	pop    %r12
    9368:	41 5d                	pop    %r13
    936a:	41 5e                	pop    %r14
    936c:	44 89 c0             	mov    %r8d,%eax
    936f:	41 5f                	pop    %r15
    9371:	c3                   	retq   
    9372:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    9378:	44 89 ea             	mov    %r13d,%edx
    937b:	be 02 00 00 00       	mov    $0x2,%esi
    9380:	44 89 e7             	mov    %r12d,%edi
    9383:	e8 88 f2 ff ff       	callq  8610 <galois_single_multiply>
    9388:	43 89 04 ae          	mov    %eax,(%r14,%r13,4)
    938c:	45 85 ed             	test   %r13d,%r13d
    938f:	7e 56                	jle    93e7 <cauchy_n_ones+0x177>
    9391:	4d 89 e8             	mov    %r13,%r8
    9394:	49 c1 e0 05          	shl    $0x5,%r8
    9398:	31 d2                	xor    %edx,%edx
    939a:	31 f6                	xor    %esi,%esi
    939c:	4c 8d 15 bd 83 00 00 	lea    0x83bd(%rip),%r10        # 11760 <ONEs>
    93a3:	4d 01 e8             	add    %r13,%r8
    93a6:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    93ac:	0f 1f 40 00          	nopl   0x0(%rax)
    93b0:	0f a3 d0             	bt     %edx,%eax
    93b3:	73 11                	jae    93c6 <cauchy_n_ones+0x156>
    93b5:	48 63 ce             	movslq %esi,%rcx
    93b8:	4c 01 c1             	add    %r8,%rcx
    93bb:	c4 c2 69 f7 f9       	shlx   %edx,%r9d,%edi
    93c0:	41 89 3c 8a          	mov    %edi,(%r10,%rcx,4)
    93c4:	ff c6                	inc    %esi
    93c6:	ff c2                	inc    %edx
    93c8:	39 d5                	cmp    %edx,%ebp
    93ca:	75 e4                	jne    93b0 <cauchy_n_ones+0x140>
    93cc:	48 8d 05 ad 94 00 00 	lea    0x94ad(%rip),%rax        # 12880 <NOs>
    93d3:	42 89 34 a8          	mov    %esi,(%rax,%r13,4)
    93d7:	85 ed                	test   %ebp,%ebp
    93d9:	0f 8f d4 fe ff ff    	jg     92b3 <cauchy_n_ones+0x43>
    93df:	45 31 c0             	xor    %r8d,%r8d
    93e2:	e9 79 ff ff ff       	jmpq   9360 <cauchy_n_ones+0xf0>
    93e7:	31 f6                	xor    %esi,%esi
    93e9:	eb e1                	jmp    93cc <cauchy_n_ones+0x15c>
    93eb:	41 89 c8             	mov    %ecx,%r8d
    93ee:	e9 6d ff ff ff       	jmpq   9360 <cauchy_n_ones+0xf0>
    93f3:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    93fa:	00 00 00 00 
    93fe:	66 90                	xchg   %ax,%ax

0000000000009400 <cauchy_original_coding_matrix>:
    9400:	f3 0f 1e fa          	endbr64 
    9404:	41 57                	push   %r15
    9406:	41 89 ff             	mov    %edi,%r15d
    9409:	41 56                	push   %r14
    940b:	41 55                	push   %r13
    940d:	41 54                	push   %r12
    940f:	55                   	push   %rbp
    9410:	89 d5                	mov    %edx,%ebp
    9412:	53                   	push   %rbx
    9413:	48 83 ec 18          	sub    $0x18,%rsp
    9417:	89 34 24             	mov    %esi,(%rsp)
    941a:	83 fa 1e             	cmp    $0x1e,%edx
    941d:	7f 15                	jg     9434 <cauchy_original_coding_matrix+0x34>
    941f:	b8 01 00 00 00       	mov    $0x1,%eax
    9424:	8d 14 37             	lea    (%rdi,%rsi,1),%edx
    9427:	c4 e2 51 f7 c0       	shlx   %ebp,%eax,%eax
    942c:	39 c2                	cmp    %eax,%edx
    942e:	0f 8f 9d 00 00 00    	jg     94d1 <cauchy_original_coding_matrix+0xd1>
    9434:	44 8b 34 24          	mov    (%rsp),%r14d
    9438:	44 89 f7             	mov    %r14d,%edi
    943b:	41 0f af ff          	imul   %r15d,%edi
    943f:	48 63 ff             	movslq %edi,%rdi
    9442:	48 c1 e7 02          	shl    $0x2,%rdi
    9446:	e8 b5 7f ff ff       	callq  1400 <malloc@plt>
    944b:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    9450:	48 85 c0             	test   %rax,%rax
    9453:	74 7c                	je     94d1 <cauchy_original_coding_matrix+0xd1>
    9455:	c7 44 24 04 00 00 00 	movl   $0x0,0x4(%rsp)
    945c:	00 
    945d:	31 db                	xor    %ebx,%ebx
    945f:	47 8d 24 3e          	lea    (%r14,%r15,1),%r12d
    9463:	45 85 f6             	test   %r14d,%r14d
    9466:	7e 55                	jle    94bd <cauchy_original_coding_matrix+0xbd>
    9468:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    946f:	00 
    9470:	45 85 ff             	test   %r15d,%r15d
    9473:	7e 41                	jle    94b6 <cauchy_original_coding_matrix+0xb6>
    9475:	48 63 44 24 04       	movslq 0x4(%rsp),%rax
    947a:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    947f:	44 8b 34 24          	mov    (%rsp),%r14d
    9483:	4c 8d 2c 81          	lea    (%rcx,%rax,4),%r13
    9487:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    948e:	00 00 
    9490:	44 89 f6             	mov    %r14d,%esi
    9493:	31 de                	xor    %ebx,%esi
    9495:	89 ea                	mov    %ebp,%edx
    9497:	bf 01 00 00 00       	mov    $0x1,%edi
    949c:	e8 8f f1 ff ff       	callq  8630 <galois_single_divide>
    94a1:	41 ff c6             	inc    %r14d
    94a4:	41 89 45 00          	mov    %eax,0x0(%r13)
    94a8:	49 83 c5 04          	add    $0x4,%r13
    94ac:	45 39 e6             	cmp    %r12d,%r14d
    94af:	75 df                	jne    9490 <cauchy_original_coding_matrix+0x90>
    94b1:	44 01 7c 24 04       	add    %r15d,0x4(%rsp)
    94b6:	ff c3                	inc    %ebx
    94b8:	39 1c 24             	cmp    %ebx,(%rsp)
    94bb:	75 b3                	jne    9470 <cauchy_original_coding_matrix+0x70>
    94bd:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    94c2:	48 83 c4 18          	add    $0x18,%rsp
    94c6:	5b                   	pop    %rbx
    94c7:	5d                   	pop    %rbp
    94c8:	41 5c                	pop    %r12
    94ca:	41 5d                	pop    %r13
    94cc:	41 5e                	pop    %r14
    94ce:	41 5f                	pop    %r15
    94d0:	c3                   	retq   
    94d1:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    94d8:	00 00 
    94da:	eb e1                	jmp    94bd <cauchy_original_coding_matrix+0xbd>
    94dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000094e0 <cauchy_xy_coding_matrix>:
    94e0:	f3 0f 1e fa          	endbr64 
    94e4:	41 57                	push   %r15
    94e6:	49 89 cf             	mov    %rcx,%r15
    94e9:	41 56                	push   %r14
    94eb:	4d 89 c6             	mov    %r8,%r14
    94ee:	41 55                	push   %r13
    94f0:	41 89 fd             	mov    %edi,%r13d
    94f3:	0f af fe             	imul   %esi,%edi
    94f6:	41 54                	push   %r12
    94f8:	41 89 f4             	mov    %esi,%r12d
    94fb:	48 63 ff             	movslq %edi,%rdi
    94fe:	55                   	push   %rbp
    94ff:	48 c1 e7 02          	shl    $0x2,%rdi
    9503:	53                   	push   %rbx
    9504:	89 d3                	mov    %edx,%ebx
    9506:	48 83 ec 28          	sub    $0x28,%rsp
    950a:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
    950f:	e8 ec 7e ff ff       	callq  1400 <malloc@plt>
    9514:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    9519:	48 85 c0             	test   %rax,%rax
    951c:	74 75                	je     9593 <cauchy_xy_coding_matrix+0xb3>
    951e:	45 85 e4             	test   %r12d,%r12d
    9521:	7e 70                	jle    9593 <cauchy_xy_coding_matrix+0xb3>
    9523:	41 8d 44 24 ff       	lea    -0x1(%r12),%eax
    9528:	49 8d 44 87 04       	lea    0x4(%r15,%rax,4),%rax
    952d:	c7 44 24 04 00 00 00 	movl   $0x0,0x4(%rsp)
    9534:	00 
    9535:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    953a:	41 8d 45 ff          	lea    -0x1(%r13),%eax
    953e:	49 8d 6c 86 04       	lea    0x4(%r14,%rax,4),%rbp
    9543:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    9548:	45 85 ed             	test   %r13d,%r13d
    954b:	7e 3b                	jle    9588 <cauchy_xy_coding_matrix+0xa8>
    954d:	48 63 44 24 04       	movslq 0x4(%rsp),%rax
    9552:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    9557:	4c 8b 74 24 18       	mov    0x18(%rsp),%r14
    955c:	4c 8d 24 81          	lea    (%rcx,%rax,4),%r12
    9560:	41 8b 37             	mov    (%r15),%esi
    9563:	89 da                	mov    %ebx,%edx
    9565:	41 33 36             	xor    (%r14),%esi
    9568:	bf 01 00 00 00       	mov    $0x1,%edi
    956d:	e8 be f0 ff ff       	callq  8630 <galois_single_divide>
    9572:	49 83 c6 04          	add    $0x4,%r14
    9576:	41 89 04 24          	mov    %eax,(%r12)
    957a:	49 83 c4 04          	add    $0x4,%r12
    957e:	49 39 ee             	cmp    %rbp,%r14
    9581:	75 dd                	jne    9560 <cauchy_xy_coding_matrix+0x80>
    9583:	44 01 6c 24 04       	add    %r13d,0x4(%rsp)
    9588:	49 83 c7 04          	add    $0x4,%r15
    958c:	4c 3b 7c 24 10       	cmp    0x10(%rsp),%r15
    9591:	75 b5                	jne    9548 <cauchy_xy_coding_matrix+0x68>
    9593:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    9598:	48 83 c4 28          	add    $0x28,%rsp
    959c:	5b                   	pop    %rbx
    959d:	5d                   	pop    %rbp
    959e:	41 5c                	pop    %r12
    95a0:	41 5d                	pop    %r13
    95a2:	41 5e                	pop    %r14
    95a4:	41 5f                	pop    %r15
    95a6:	c3                   	retq   
    95a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    95ae:	00 00 

00000000000095b0 <cauchy_improve_coding_matrix>:
    95b0:	f3 0f 1e fa          	endbr64 
    95b4:	41 57                	push   %r15
    95b6:	41 56                	push   %r14
    95b8:	41 55                	push   %r13
    95ba:	41 54                	push   %r12
    95bc:	55                   	push   %rbp
    95bd:	53                   	push   %rbx
    95be:	89 d3                	mov    %edx,%ebx
    95c0:	48 83 ec 48          	sub    $0x48,%rsp
    95c4:	89 7c 24 30          	mov    %edi,0x30(%rsp)
    95c8:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
    95cc:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
    95d1:	85 ff                	test   %edi,%edi
    95d3:	0f 8e 7e 00 00 00    	jle    9657 <cauchy_improve_coding_matrix+0xa7>
    95d9:	8d 47 ff             	lea    -0x1(%rdi),%eax
    95dc:	4c 63 ef             	movslq %edi,%r13
    95df:	48 89 04 24          	mov    %rax,(%rsp)
    95e3:	49 c1 e5 02          	shl    $0x2,%r13
    95e7:	31 ed                	xor    %ebp,%ebp
    95e9:	eb 12                	jmp    95fd <cauchy_improve_coding_matrix+0x4d>
    95eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    95f0:	48 8d 45 01          	lea    0x1(%rbp),%rax
    95f4:	48 3b 2c 24          	cmp    (%rsp),%rbp
    95f8:	74 5d                	je     9657 <cauchy_improve_coding_matrix+0xa7>
    95fa:	48 89 c5             	mov    %rax,%rbp
    95fd:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9602:	8b 34 a8             	mov    (%rax,%rbp,4),%esi
    9605:	83 fe 01             	cmp    $0x1,%esi
    9608:	74 e6                	je     95f0 <cauchy_improve_coding_matrix+0x40>
    960a:	89 da                	mov    %ebx,%edx
    960c:	bf 01 00 00 00       	mov    $0x1,%edi
    9611:	e8 1a f0 ff ff       	callq  8630 <galois_single_divide>
    9616:	8b 54 24 1c          	mov    0x1c(%rsp),%edx
    961a:	41 89 c4             	mov    %eax,%r12d
    961d:	85 d2                	test   %edx,%edx
    961f:	7e cf                	jle    95f0 <cauchy_improve_coding_matrix+0x40>
    9621:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9626:	45 31 ff             	xor    %r15d,%r15d
    9629:	4c 8d 34 a8          	lea    (%rax,%rbp,4),%r14
    962d:	0f 1f 00             	nopl   (%rax)
    9630:	41 8b 3e             	mov    (%r14),%edi
    9633:	89 da                	mov    %ebx,%edx
    9635:	44 89 e6             	mov    %r12d,%esi
    9638:	e8 d3 ef ff ff       	callq  8610 <galois_single_multiply>
    963d:	41 ff c7             	inc    %r15d
    9640:	41 89 06             	mov    %eax,(%r14)
    9643:	4d 01 ee             	add    %r13,%r14
    9646:	44 39 7c 24 1c       	cmp    %r15d,0x1c(%rsp)
    964b:	75 e3                	jne    9630 <cauchy_improve_coding_matrix+0x80>
    964d:	48 8d 45 01          	lea    0x1(%rbp),%rax
    9651:	48 3b 2c 24          	cmp    (%rsp),%rbp
    9655:	75 a3                	jne    95fa <cauchy_improve_coding_matrix+0x4a>
    9657:	83 7c 24 1c 01       	cmpl   $0x1,0x1c(%rsp)
    965c:	0f 8e 3b 01 00 00    	jle    979d <cauchy_improve_coding_matrix+0x1ed>
    9662:	48 63 44 24 30       	movslq 0x30(%rsp),%rax
    9667:	c7 44 24 20 01 00 00 	movl   $0x1,0x20(%rsp)
    966e:	00 
    966f:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    9676:	00 
    9677:	48 89 d7             	mov    %rdx,%rdi
    967a:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
    967f:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    9684:	48 89 c1             	mov    %rax,%rcx
    9687:	48 89 d6             	mov    %rdx,%rsi
    968a:	48 01 fe             	add    %rdi,%rsi
    968d:	48 89 34 24          	mov    %rsi,(%rsp)
    9691:	8d 70 ff             	lea    -0x1(%rax),%esi
    9694:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
    9699:	89 4c 24 34          	mov    %ecx,0x34(%rsp)
    969d:	48 01 f0             	add    %rsi,%rax
    96a0:	4c 8d 7c 82 04       	lea    0x4(%rdx,%rax,4),%r15
    96a5:	0f 1f 00             	nopl   (%rax)
    96a8:	8b 44 24 30          	mov    0x30(%rsp),%eax
    96ac:	85 c0                	test   %eax,%eax
    96ae:	0f 8e c3 00 00 00    	jle    9777 <cauchy_improve_coding_matrix+0x1c7>
    96b4:	48 8b 2c 24          	mov    (%rsp),%rbp
    96b8:	45 31 e4             	xor    %r12d,%r12d
    96bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    96c0:	8b 7d 00             	mov    0x0(%rbp),%edi
    96c3:	89 de                	mov    %ebx,%esi
    96c5:	e8 a6 fb ff ff       	callq  9270 <cauchy_n_ones>
    96ca:	48 83 c5 04          	add    $0x4,%rbp
    96ce:	41 01 c4             	add    %eax,%r12d
    96d1:	4c 39 fd             	cmp    %r15,%rbp
    96d4:	75 ea                	jne    96c0 <cauchy_improve_coding_matrix+0x110>
    96d6:	44 89 64 24 18       	mov    %r12d,0x18(%rsp)
    96db:	c7 44 24 24 ff ff ff 	movl   $0xffffffff,0x24(%rsp)
    96e2:	ff 
    96e3:	45 31 f6             	xor    %r14d,%r14d
    96e6:	eb 16                	jmp    96fe <cauchy_improve_coding_matrix+0x14e>
    96e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    96ef:	00 
    96f0:	49 8d 46 01          	lea    0x1(%r14),%rax
    96f4:	4c 3b 74 24 10       	cmp    0x10(%rsp),%r14
    96f9:	74 75                	je     9770 <cauchy_improve_coding_matrix+0x1c0>
    96fb:	49 89 c6             	mov    %rax,%r14
    96fe:	48 8b 04 24          	mov    (%rsp),%rax
    9702:	44 89 74 24 0c       	mov    %r14d,0xc(%rsp)
    9707:	42 8b 34 b0          	mov    (%rax,%r14,4),%esi
    970b:	83 fe 01             	cmp    $0x1,%esi
    970e:	74 e0                	je     96f0 <cauchy_improve_coding_matrix+0x140>
    9710:	89 da                	mov    %ebx,%edx
    9712:	bf 01 00 00 00       	mov    $0x1,%edi
    9717:	e8 14 ef ff ff       	callq  8630 <galois_single_divide>
    971c:	4c 8b 2c 24          	mov    (%rsp),%r13
    9720:	41 89 c4             	mov    %eax,%r12d
    9723:	31 ed                	xor    %ebp,%ebp
    9725:	0f 1f 00             	nopl   (%rax)
    9728:	41 8b 7d 00          	mov    0x0(%r13),%edi
    972c:	89 da                	mov    %ebx,%edx
    972e:	44 89 e6             	mov    %r12d,%esi
    9731:	e8 da ee ff ff       	callq  8610 <galois_single_multiply>
    9736:	89 c7                	mov    %eax,%edi
    9738:	89 de                	mov    %ebx,%esi
    973a:	e8 31 fb ff ff       	callq  9270 <cauchy_n_ones>
    973f:	49 83 c5 04          	add    $0x4,%r13
    9743:	01 c5                	add    %eax,%ebp
    9745:	4d 39 fd             	cmp    %r15,%r13
    9748:	75 de                	jne    9728 <cauchy_improve_coding_matrix+0x178>
    974a:	3b 6c 24 18          	cmp    0x18(%rsp),%ebp
    974e:	7d a0                	jge    96f0 <cauchy_improve_coding_matrix+0x140>
    9750:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    9754:	89 6c 24 18          	mov    %ebp,0x18(%rsp)
    9758:	89 44 24 24          	mov    %eax,0x24(%rsp)
    975c:	49 8d 46 01          	lea    0x1(%r14),%rax
    9760:	4c 3b 74 24 10       	cmp    0x10(%rsp),%r14
    9765:	75 94                	jne    96fb <cauchy_improve_coding_matrix+0x14b>
    9767:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    976e:	00 00 
    9770:	83 7c 24 24 ff       	cmpl   $0xffffffff,0x24(%rsp)
    9775:	75 39                	jne    97b0 <cauchy_improve_coding_matrix+0x200>
    9777:	ff 44 24 20          	incl   0x20(%rsp)
    977b:	48 8b 54 24 38       	mov    0x38(%rsp),%rdx
    9780:	8b 74 24 30          	mov    0x30(%rsp),%esi
    9784:	48 01 14 24          	add    %rdx,(%rsp)
    9788:	8b 44 24 20          	mov    0x20(%rsp),%eax
    978c:	01 74 24 34          	add    %esi,0x34(%rsp)
    9790:	49 01 d7             	add    %rdx,%r15
    9793:	39 44 24 1c          	cmp    %eax,0x1c(%rsp)
    9797:	0f 85 0b ff ff ff    	jne    96a8 <cauchy_improve_coding_matrix+0xf8>
    979d:	48 83 c4 48          	add    $0x48,%rsp
    97a1:	5b                   	pop    %rbx
    97a2:	5d                   	pop    %rbp
    97a3:	41 5c                	pop    %r12
    97a5:	41 5d                	pop    %r13
    97a7:	41 5e                	pop    %r14
    97a9:	41 5f                	pop    %r15
    97ab:	c3                   	retq   
    97ac:	0f 1f 40 00          	nopl   0x0(%rax)
    97b0:	8b 44 24 24          	mov    0x24(%rsp),%eax
    97b4:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
    97b9:	03 44 24 34          	add    0x34(%rsp),%eax
    97bd:	48 98                	cltq   
    97bf:	8b 34 81             	mov    (%rcx,%rax,4),%esi
    97c2:	89 da                	mov    %ebx,%edx
    97c4:	bf 01 00 00 00       	mov    $0x1,%edi
    97c9:	e8 62 ee ff ff       	callq  8630 <galois_single_divide>
    97ce:	4c 8b 24 24          	mov    (%rsp),%r12
    97d2:	89 c5                	mov    %eax,%ebp
    97d4:	0f 1f 40 00          	nopl   0x0(%rax)
    97d8:	41 8b 3c 24          	mov    (%r12),%edi
    97dc:	89 da                	mov    %ebx,%edx
    97de:	89 ee                	mov    %ebp,%esi
    97e0:	e8 2b ee ff ff       	callq  8610 <galois_single_multiply>
    97e5:	41 89 04 24          	mov    %eax,(%r12)
    97e9:	49 83 c4 04          	add    $0x4,%r12
    97ed:	4d 39 fc             	cmp    %r15,%r12
    97f0:	75 e6                	jne    97d8 <cauchy_improve_coding_matrix+0x228>
    97f2:	eb 83                	jmp    9777 <cauchy_improve_coding_matrix+0x1c7>
    97f4:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    97fb:	00 00 00 00 
    97ff:	90                   	nop

0000000000009800 <cauchy_good_general_coding_matrix>:
    9800:	f3 0f 1e fa          	endbr64 
    9804:	41 56                	push   %r14
    9806:	41 89 fe             	mov    %edi,%r14d
    9809:	41 55                	push   %r13
    980b:	41 89 d5             	mov    %edx,%r13d
    980e:	41 54                	push   %r12
    9810:	55                   	push   %rbp
    9811:	89 f5                	mov    %esi,%ebp
    9813:	53                   	push   %rbx
    9814:	83 fe 02             	cmp    $0x2,%esi
    9817:	75 0f                	jne    9828 <cauchy_good_general_coding_matrix+0x28>
    9819:	48 63 da             	movslq %edx,%rbx
    981c:	48 8d 05 9d 15 00 00 	lea    0x159d(%rip),%rax        # adc0 <cbest_max_k>
    9823:	39 3c 98             	cmp    %edi,(%rax,%rbx,4)
    9826:	7d 38                	jge    9860 <cauchy_good_general_coding_matrix+0x60>
    9828:	44 89 ea             	mov    %r13d,%edx
    982b:	89 ee                	mov    %ebp,%esi
    982d:	44 89 f7             	mov    %r14d,%edi
    9830:	e8 cb fb ff ff       	callq  9400 <cauchy_original_coding_matrix>
    9835:	49 89 c4             	mov    %rax,%r12
    9838:	48 85 c0             	test   %rax,%rax
    983b:	74 10                	je     984d <cauchy_good_general_coding_matrix+0x4d>
    983d:	48 89 c1             	mov    %rax,%rcx
    9840:	44 89 ea             	mov    %r13d,%edx
    9843:	89 ee                	mov    %ebp,%esi
    9845:	44 89 f7             	mov    %r14d,%edi
    9848:	e8 63 fd ff ff       	callq  95b0 <cauchy_improve_coding_matrix>
    984d:	5b                   	pop    %rbx
    984e:	5d                   	pop    %rbp
    984f:	4c 89 e0             	mov    %r12,%rax
    9852:	41 5c                	pop    %r12
    9854:	41 5d                	pop    %r13
    9856:	41 5e                	pop    %r14
    9858:	c3                   	retq   
    9859:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    9860:	8d 3c 3f             	lea    (%rdi,%rdi,1),%edi
    9863:	48 63 ff             	movslq %edi,%rdi
    9866:	48 c1 e7 02          	shl    $0x2,%rdi
    986a:	e8 91 7b ff ff       	callq  1400 <malloc@plt>
    986f:	49 89 c4             	mov    %rax,%r12
    9872:	48 85 c0             	test   %rax,%rax
    9875:	74 d6                	je     984d <cauchy_good_general_coding_matrix+0x4d>
    9877:	8b 05 cb 7e 00 00    	mov    0x7ecb(%rip),%eax        # 11748 <cbest_init>
    987d:	85 c0                	test   %eax,%eax
    987f:	0f 85 93 01 00 00    	jne    9a18 <cauchy_good_general_coding_matrix+0x218>
    9885:	48 8d 05 f4 77 00 00 	lea    0x77f4(%rip),%rax        # 11080 <cbest_2>
    988c:	48 89 05 bd 7d 00 00 	mov    %rax,0x7dbd(%rip)        # 11650 <cbest_all+0x10>
    9893:	48 8d 05 c6 77 00 00 	lea    0x77c6(%rip),%rax        # 11060 <cbest_3>
    989a:	48 89 05 b7 7d 00 00 	mov    %rax,0x7db7(%rip)        # 11658 <cbest_all+0x18>
    98a1:	48 8d 05 78 77 00 00 	lea    0x7778(%rip),%rax        # 11020 <cbest_4>
    98a8:	48 89 05 b1 7d 00 00 	mov    %rax,0x7db1(%rip)        # 11660 <cbest_all+0x20>
    98af:	48 8d 05 ea 76 00 00 	lea    0x76ea(%rip),%rax        # 10fa0 <cbest_5>
    98b6:	48 89 05 ab 7d 00 00 	mov    %rax,0x7dab(%rip)        # 11668 <cbest_all+0x28>
    98bd:	48 8d 05 dc 75 00 00 	lea    0x75dc(%rip),%rax        # 10ea0 <cbest_6>
    98c4:	48 89 05 a5 7d 00 00 	mov    %rax,0x7da5(%rip)        # 11670 <cbest_all+0x30>
    98cb:	48 8d 05 ce 73 00 00 	lea    0x73ce(%rip),%rax        # 10ca0 <cbest_7>
    98d2:	48 89 05 9f 7d 00 00 	mov    %rax,0x7d9f(%rip)        # 11678 <cbest_all+0x38>
    98d9:	48 8d 05 c0 6f 00 00 	lea    0x6fc0(%rip),%rax        # 108a0 <cbest_8>
    98e0:	48 89 05 99 7d 00 00 	mov    %rax,0x7d99(%rip)        # 11680 <cbest_all+0x40>
    98e7:	48 8d 05 b2 67 00 00 	lea    0x67b2(%rip),%rax        # 100a0 <cbest_9>
    98ee:	48 89 05 93 7d 00 00 	mov    %rax,0x7d93(%rip)        # 11688 <cbest_all+0x48>
    98f5:	48 8d 05 a4 57 00 00 	lea    0x57a4(%rip),%rax        # f0a0 <cbest_10>
    98fc:	48 89 05 8d 7d 00 00 	mov    %rax,0x7d8d(%rip)        # 11690 <cbest_all+0x50>
    9903:	48 8d 05 96 47 00 00 	lea    0x4796(%rip),%rax        # e0a0 <cbest_11>
    990a:	c7 05 34 7e 00 00 01 	movl   $0x1,0x7e34(%rip)        # 11748 <cbest_init>
    9911:	00 00 00 
    9914:	48 c7 05 21 7d 00 00 	movq   $0x0,0x7d21(%rip)        # 11640 <cbest_all>
    991b:	00 00 00 00 
    991f:	48 c7 05 1e 7d 00 00 	movq   $0x0,0x7d1e(%rip)        # 11648 <cbest_all+0x8>
    9926:	00 00 00 00 
    992a:	48 89 05 67 7d 00 00 	mov    %rax,0x7d67(%rip)        # 11698 <cbest_all+0x58>
    9931:	48 c7 05 64 7d 00 00 	movq   $0x0,0x7d64(%rip)        # 116a0 <cbest_all+0x60>
    9938:	00 00 00 00 
    993c:	48 c7 05 61 7d 00 00 	movq   $0x0,0x7d61(%rip)        # 116a8 <cbest_all+0x68>
    9943:	00 00 00 00 
    9947:	48 c7 05 5e 7d 00 00 	movq   $0x0,0x7d5e(%rip)        # 116b0 <cbest_all+0x70>
    994e:	00 00 00 00 
    9952:	48 c7 05 5b 7d 00 00 	movq   $0x0,0x7d5b(%rip)        # 116b8 <cbest_all+0x78>
    9959:	00 00 00 00 
    995d:	48 c7 05 58 7d 00 00 	movq   $0x0,0x7d58(%rip)        # 116c0 <cbest_all+0x80>
    9964:	00 00 00 00 
    9968:	48 c7 05 55 7d 00 00 	movq   $0x0,0x7d55(%rip)        # 116c8 <cbest_all+0x88>
    996f:	00 00 00 00 
    9973:	48 c7 05 52 7d 00 00 	movq   $0x0,0x7d52(%rip)        # 116d0 <cbest_all+0x90>
    997a:	00 00 00 00 
    997e:	48 c7 05 4f 7d 00 00 	movq   $0x0,0x7d4f(%rip)        # 116d8 <cbest_all+0x98>
    9985:	00 00 00 00 
    9989:	48 c7 05 4c 7d 00 00 	movq   $0x0,0x7d4c(%rip)        # 116e0 <cbest_all+0xa0>
    9990:	00 00 00 00 
    9994:	48 c7 05 49 7d 00 00 	movq   $0x0,0x7d49(%rip)        # 116e8 <cbest_all+0xa8>
    999b:	00 00 00 00 
    999f:	48 c7 05 46 7d 00 00 	movq   $0x0,0x7d46(%rip)        # 116f0 <cbest_all+0xb0>
    99a6:	00 00 00 00 
    99aa:	48 c7 05 43 7d 00 00 	movq   $0x0,0x7d43(%rip)        # 116f8 <cbest_all+0xb8>
    99b1:	00 00 00 00 
    99b5:	48 c7 05 40 7d 00 00 	movq   $0x0,0x7d40(%rip)        # 11700 <cbest_all+0xc0>
    99bc:	00 00 00 00 
    99c0:	48 c7 05 3d 7d 00 00 	movq   $0x0,0x7d3d(%rip)        # 11708 <cbest_all+0xc8>
    99c7:	00 00 00 00 
    99cb:	48 c7 05 3a 7d 00 00 	movq   $0x0,0x7d3a(%rip)        # 11710 <cbest_all+0xd0>
    99d2:	00 00 00 00 
    99d6:	48 c7 05 37 7d 00 00 	movq   $0x0,0x7d37(%rip)        # 11718 <cbest_all+0xd8>
    99dd:	00 00 00 00 
    99e1:	48 c7 05 34 7d 00 00 	movq   $0x0,0x7d34(%rip)        # 11720 <cbest_all+0xe0>
    99e8:	00 00 00 00 
    99ec:	48 c7 05 31 7d 00 00 	movq   $0x0,0x7d31(%rip)        # 11728 <cbest_all+0xe8>
    99f3:	00 00 00 00 
    99f7:	48 c7 05 2e 7d 00 00 	movq   $0x0,0x7d2e(%rip)        # 11730 <cbest_all+0xf0>
    99fe:	00 00 00 00 
    9a02:	48 c7 05 2b 7d 00 00 	movq   $0x0,0x7d2b(%rip)        # 11738 <cbest_all+0xf8>
    9a09:	00 00 00 00 
    9a0d:	48 c7 05 28 7d 00 00 	movq   $0x0,0x7d28(%rip)        # 11740 <cbest_all+0x100>
    9a14:	00 00 00 00 
    9a18:	45 85 f6             	test   %r14d,%r14d
    9a1b:	0f 8e 2c fe ff ff    	jle    984d <cauchy_good_general_coding_matrix+0x4d>
    9a21:	48 8d 05 18 7c 00 00 	lea    0x7c18(%rip),%rax        # 11640 <cbest_all>
    9a28:	49 63 fe             	movslq %r14d,%rdi
    9a2b:	4c 8b 04 d8          	mov    (%rax,%rbx,8),%r8
    9a2f:	41 8d 76 ff          	lea    -0x1(%r14),%esi
    9a33:	49 8d 0c bc          	lea    (%r12,%rdi,4),%rcx
    9a37:	31 d2                	xor    %edx,%edx
    9a39:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    9a40:	41 c7 04 94 01 00 00 	movl   $0x1,(%r12,%rdx,4)
    9a47:	00 
    9a48:	41 8b 04 90          	mov    (%r8,%rdx,4),%eax
    9a4c:	89 04 91             	mov    %eax,(%rcx,%rdx,4)
    9a4f:	48 89 d0             	mov    %rdx,%rax
    9a52:	48 ff c2             	inc    %rdx
    9a55:	48 39 c6             	cmp    %rax,%rsi
    9a58:	75 e6                	jne    9a40 <cauchy_good_general_coding_matrix+0x240>
    9a5a:	5b                   	pop    %rbx
    9a5b:	5d                   	pop    %rbp
    9a5c:	4c 89 e0             	mov    %r12,%rax
    9a5f:	41 5c                	pop    %r12
    9a61:	41 5d                	pop    %r13
    9a63:	41 5e                	pop    %r14
    9a65:	c3                   	retq   
    9a66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    9a6d:	00 00 00 

0000000000009a70 <timing_delta>:
    9a70:	f3 0f 1e fa          	endbr64 
    9a74:	41 57                	push   %r15
    9a76:	49 89 f7             	mov    %rsi,%r15
    9a79:	41 56                	push   %r14
    9a7b:	41 55                	push   %r13
    9a7d:	41 54                	push   %r12
    9a7f:	55                   	push   %rbp
    9a80:	53                   	push   %rbx
    9a81:	48 83 ec 68          	sub    $0x68,%rsp
    9a85:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
    9a8a:	c5 fb 10 05 76 8e 00 	vmovsd 0x8e76(%rip),%xmm0        # 12908 <timing_time.2619>
    9a91:	00 
    9a92:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    9a99:	00 00 
    9a9b:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    9aa0:	31 c0                	xor    %eax,%eax
    9aa2:	c5 f9 2e 05 36 0c 00 	vucomisd 0xc36(%rip),%xmm0        # a6e0 <__PRETTY_FUNCTION__.5741+0x7>
    9aa9:	00 
    9aaa:	7a 73                	jp     9b1f <timing_delta+0xaf>
    9aac:	75 71                	jne    9b1f <timing_delta+0xaf>
    9aae:	bb 10 27 00 00       	mov    $0x2710,%ebx
    9ab3:	4c 8d 74 24 10       	lea    0x10(%rsp),%r14
    9ab8:	4c 8d 6c 24 30       	lea    0x30(%rsp),%r13
    9abd:	4c 8d 64 24 40       	lea    0x40(%rsp),%r12
    9ac2:	48 8d 6c 24 20       	lea    0x20(%rsp),%rbp
    9ac7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    9ace:	00 00 
    9ad0:	4c 89 f6             	mov    %r14,%rsi
    9ad3:	31 ff                	xor    %edi,%edi
    9ad5:	e8 06 78 ff ff       	callq  12e0 <clock_gettime@plt>
    9ada:	4c 89 ee             	mov    %r13,%rsi
    9add:	31 ff                	xor    %edi,%edi
    9adf:	e8 fc 77 ff ff       	callq  12e0 <clock_gettime@plt>
    9ae4:	4c 89 e6             	mov    %r12,%rsi
    9ae7:	31 ff                	xor    %edi,%edi
    9ae9:	e8 f2 77 ff ff       	callq  12e0 <clock_gettime@plt>
    9aee:	48 89 ee             	mov    %rbp,%rsi
    9af1:	31 ff                	xor    %edi,%edi
    9af3:	e8 e8 77 ff ff       	callq  12e0 <clock_gettime@plt>
    9af8:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9afd:	c5 e1 57 db          	vxorpd %xmm3,%xmm3,%xmm3
    9b01:	48 2b 44 24 18       	sub    0x18(%rsp),%rax
    9b06:	c4 e1 e3 2a c0       	vcvtsi2sd %rax,%xmm3,%xmm0
    9b0b:	c5 fb 5e 05 35 13 00 	vdivsd 0x1335(%rip),%xmm0,%xmm0        # ae48 <cbest_max_k+0x88>
    9b12:	00 
    9b13:	c5 fb 11 05 ed 8d 00 	vmovsd %xmm0,0x8ded(%rip)        # 12908 <timing_time.2619>
    9b1a:	00 
    9b1b:	ff cb                	dec    %ebx
    9b1d:	75 b1                	jne    9ad0 <timing_delta+0x60>
    9b1f:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
    9b24:	49 8b 47 08          	mov    0x8(%r15),%rax
    9b28:	c5 d9 57 e4          	vxorpd %xmm4,%xmm4,%xmm4
    9b2c:	48 2b 42 08          	sub    0x8(%rdx),%rax
    9b30:	c4 e1 db 2a c8       	vcvtsi2sd %rax,%xmm4,%xmm1
    9b35:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
    9b3a:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    9b41:	00 00 
    9b43:	c5 f3 5c c0          	vsubsd %xmm0,%xmm1,%xmm0
    9b47:	c4 c1 db 2a 0f       	vcvtsi2sdq (%r15),%xmm4,%xmm1
    9b4c:	c5 fb 5e 05 fc 12 00 	vdivsd 0x12fc(%rip),%xmm0,%xmm0        # ae50 <cbest_max_k+0x90>
    9b53:	00 
    9b54:	c5 f9 28 d1          	vmovapd %xmm1,%xmm2
    9b58:	c4 e1 db 2a 0a       	vcvtsi2sdq (%rdx),%xmm4,%xmm1
    9b5d:	c5 eb 5c c9          	vsubsd %xmm1,%xmm2,%xmm1
    9b61:	c5 fb 58 c1          	vaddsd %xmm1,%xmm0,%xmm0
    9b65:	75 0f                	jne    9b76 <timing_delta+0x106>
    9b67:	48 83 c4 68          	add    $0x68,%rsp
    9b6b:	5b                   	pop    %rbx
    9b6c:	5d                   	pop    %rbp
    9b6d:	41 5c                	pop    %r12
    9b6f:	41 5d                	pop    %r13
    9b71:	41 5e                	pop    %r14
    9b73:	41 5f                	pop    %r15
    9b75:	c3                   	retq   
    9b76:	e8 95 77 ff ff       	callq  1310 <__stack_chk_fail@plt>
    9b7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000009b80 <__libc_csu_init>:
    9b80:	f3 0f 1e fa          	endbr64 
    9b84:	41 57                	push   %r15
    9b86:	4c 8d 3d 13 41 00 00 	lea    0x4113(%rip),%r15        # dca0 <__frame_dummy_init_array_entry>
    9b8d:	41 56                	push   %r14
    9b8f:	49 89 d6             	mov    %rdx,%r14
    9b92:	41 55                	push   %r13
    9b94:	49 89 f5             	mov    %rsi,%r13
    9b97:	41 54                	push   %r12
    9b99:	41 89 fc             	mov    %edi,%r12d
    9b9c:	55                   	push   %rbp
    9b9d:	48 8d 2d 04 41 00 00 	lea    0x4104(%rip),%rbp        # dca8 <__do_global_dtors_aux_fini_array_entry>
    9ba4:	53                   	push   %rbx
    9ba5:	4c 29 fd             	sub    %r15,%rbp
    9ba8:	48 83 ec 08          	sub    $0x8,%rsp
    9bac:	e8 4f 74 ff ff       	callq  1000 <_init>
    9bb1:	48 c1 fd 03          	sar    $0x3,%rbp
    9bb5:	74 1f                	je     9bd6 <__libc_csu_init+0x56>
    9bb7:	31 db                	xor    %ebx,%ebx
    9bb9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    9bc0:	4c 89 f2             	mov    %r14,%rdx
    9bc3:	4c 89 ee             	mov    %r13,%rsi
    9bc6:	44 89 e7             	mov    %r12d,%edi
    9bc9:	41 ff 14 df          	callq  *(%r15,%rbx,8)
    9bcd:	48 83 c3 01          	add    $0x1,%rbx
    9bd1:	48 39 dd             	cmp    %rbx,%rbp
    9bd4:	75 ea                	jne    9bc0 <__libc_csu_init+0x40>
    9bd6:	48 83 c4 08          	add    $0x8,%rsp
    9bda:	5b                   	pop    %rbx
    9bdb:	5d                   	pop    %rbp
    9bdc:	41 5c                	pop    %r12
    9bde:	41 5d                	pop    %r13
    9be0:	41 5e                	pop    %r14
    9be2:	41 5f                	pop    %r15
    9be4:	c3                   	retq   
    9be5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    9bec:	00 00 00 00 

0000000000009bf0 <__libc_csu_fini>:
    9bf0:	f3 0f 1e fa          	endbr64 
    9bf4:	c3                   	retq   

Disassembly of section .fini:

0000000000009bf8 <_fini>:
    9bf8:	f3 0f 1e fa          	endbr64 
    9bfc:	48 83 ec 08          	sub    $0x8,%rsp
    9c00:	48 83 c4 08          	add    $0x8,%rsp
    9c04:	c3                   	retq   

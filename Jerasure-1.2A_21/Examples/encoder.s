
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
    1020:	ff 35 8a ce 00 00    	pushq  0xce8a(%rip)        # deb0 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	f2 ff 25 8b ce 00 00 	bnd jmpq *0xce8b(%rip)        # deb8 <_GLOBAL_OFFSET_TABLE_+0x10>
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

Disassembly of section .plt.got:

0000000000001260 <__cxa_finalize@plt>:
    1260:	f3 0f 1e fa          	endbr64 
    1264:	f2 ff 25 8d cd 00 00 	bnd jmpq *0xcd8d(%rip)        # dff8 <__cxa_finalize@GLIBC_2.2.5>
    126b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .plt.sec:

0000000000001270 <free@plt>:
    1270:	f3 0f 1e fa          	endbr64 
    1274:	f2 ff 25 45 cc 00 00 	bnd jmpq *0xcc45(%rip)        # dec0 <free@GLIBC_2.2.5>
    127b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001280 <putchar@plt>:
    1280:	f3 0f 1e fa          	endbr64 
    1284:	f2 ff 25 3d cc 00 00 	bnd jmpq *0xcc3d(%rip)        # dec8 <putchar@GLIBC_2.2.5>
    128b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001290 <__errno_location@plt>:
    1290:	f3 0f 1e fa          	endbr64 
    1294:	f2 ff 25 35 cc 00 00 	bnd jmpq *0xcc35(%rip)        # ded0 <__errno_location@GLIBC_2.2.5>
    129b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012a0 <strcpy@plt>:
    12a0:	f3 0f 1e fa          	endbr64 
    12a4:	f2 ff 25 2d cc 00 00 	bnd jmpq *0xcc2d(%rip)        # ded8 <strcpy@GLIBC_2.2.5>
    12ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012b0 <mkdir@plt>:
    12b0:	f3 0f 1e fa          	endbr64 
    12b4:	f2 ff 25 25 cc 00 00 	bnd jmpq *0xcc25(%rip)        # dee0 <mkdir@GLIBC_2.2.5>
    12bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012c0 <fread@plt>:
    12c0:	f3 0f 1e fa          	endbr64 
    12c4:	f2 ff 25 1d cc 00 00 	bnd jmpq *0xcc1d(%rip)        # dee8 <fread@GLIBC_2.2.5>
    12cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012d0 <clock_gettime@plt>:
    12d0:	f3 0f 1e fa          	endbr64 
    12d4:	f2 ff 25 15 cc 00 00 	bnd jmpq *0xcc15(%rip)        # def0 <clock_gettime@GLIBC_2.17>
    12db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012e0 <fclose@plt>:
    12e0:	f3 0f 1e fa          	endbr64 
    12e4:	f2 ff 25 0d cc 00 00 	bnd jmpq *0xcc0d(%rip)        # def8 <fclose@GLIBC_2.2.5>
    12eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000012f0 <ctime@plt>:
    12f0:	f3 0f 1e fa          	endbr64 
    12f4:	f2 ff 25 05 cc 00 00 	bnd jmpq *0xcc05(%rip)        # df00 <ctime@GLIBC_2.2.5>
    12fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001300 <__stack_chk_fail@plt>:
    1300:	f3 0f 1e fa          	endbr64 
    1304:	f2 ff 25 fd cb 00 00 	bnd jmpq *0xcbfd(%rip)        # df08 <__stack_chk_fail@GLIBC_2.4>
    130b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001310 <strchr@plt>:
    1310:	f3 0f 1e fa          	endbr64 
    1314:	f2 ff 25 f5 cb 00 00 	bnd jmpq *0xcbf5(%rip)        # df10 <strchr@GLIBC_2.2.5>
    131b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001320 <strrchr@plt>:
    1320:	f3 0f 1e fa          	endbr64 
    1324:	f2 ff 25 ed cb 00 00 	bnd jmpq *0xcbed(%rip)        # df18 <strrchr@GLIBC_2.2.5>
    132b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001330 <__assert_fail@plt>:
    1330:	f3 0f 1e fa          	endbr64 
    1334:	f2 ff 25 e5 cb 00 00 	bnd jmpq *0xcbe5(%rip)        # df20 <__assert_fail@GLIBC_2.2.5>
    133b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001340 <memset@plt>:
    1340:	f3 0f 1e fa          	endbr64 
    1344:	f2 ff 25 dd cb 00 00 	bnd jmpq *0xcbdd(%rip)        # df28 <memset@GLIBC_2.2.5>
    134b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001350 <getcwd@plt>:
    1350:	f3 0f 1e fa          	endbr64 
    1354:	f2 ff 25 d5 cb 00 00 	bnd jmpq *0xcbd5(%rip)        # df30 <getcwd@GLIBC_2.2.5>
    135b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001360 <mrand48@plt>:
    1360:	f3 0f 1e fa          	endbr64 
    1364:	f2 ff 25 cd cb 00 00 	bnd jmpq *0xcbcd(%rip)        # df38 <mrand48@GLIBC_2.2.5>
    136b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001370 <calloc@plt>:
    1370:	f3 0f 1e fa          	endbr64 
    1374:	f2 ff 25 c5 cb 00 00 	bnd jmpq *0xcbc5(%rip)        # df40 <calloc@GLIBC_2.2.5>
    137b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001380 <strcmp@plt>:
    1380:	f3 0f 1e fa          	endbr64 
    1384:	f2 ff 25 bd cb 00 00 	bnd jmpq *0xcbbd(%rip)        # df48 <strcmp@GLIBC_2.2.5>
    138b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001390 <signal@plt>:
    1390:	f3 0f 1e fa          	endbr64 
    1394:	f2 ff 25 b5 cb 00 00 	bnd jmpq *0xcbb5(%rip)        # df50 <signal@GLIBC_2.2.5>
    139b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000013a0 <__memcpy_chk@plt>:
    13a0:	f3 0f 1e fa          	endbr64 
    13a4:	f2 ff 25 ad cb 00 00 	bnd jmpq *0xcbad(%rip)        # df58 <__memcpy_chk@GLIBC_2.3.4>
    13ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000013b0 <memcpy@plt>:
    13b0:	f3 0f 1e fa          	endbr64 
    13b4:	f2 ff 25 a5 cb 00 00 	bnd jmpq *0xcba5(%rip)        # df60 <memcpy@GLIBC_2.14>
    13bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000013c0 <time@plt>:
    13c0:	f3 0f 1e fa          	endbr64 
    13c4:	f2 ff 25 9d cb 00 00 	bnd jmpq *0xcb9d(%rip)        # df68 <time@GLIBC_2.2.5>
    13cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000013d0 <__stpcpy_chk@plt>:
    13d0:	f3 0f 1e fa          	endbr64 
    13d4:	f2 ff 25 95 cb 00 00 	bnd jmpq *0xcb95(%rip)        # df70 <__stpcpy_chk@GLIBC_2.3.4>
    13db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000013e0 <__xstat@plt>:
    13e0:	f3 0f 1e fa          	endbr64 
    13e4:	f2 ff 25 8d cb 00 00 	bnd jmpq *0xcb8d(%rip)        # df78 <__xstat@GLIBC_2.2.5>
    13eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000013f0 <malloc@plt>:
    13f0:	f3 0f 1e fa          	endbr64 
    13f4:	f2 ff 25 85 cb 00 00 	bnd jmpq *0xcb85(%rip)        # df80 <malloc@GLIBC_2.2.5>
    13fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001400 <__isoc99_sscanf@plt>:
    1400:	f3 0f 1e fa          	endbr64 
    1404:	f2 ff 25 7d cb 00 00 	bnd jmpq *0xcb7d(%rip)        # df88 <__isoc99_sscanf@GLIBC_2.7>
    140b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001410 <srand48@plt>:
    1410:	f3 0f 1e fa          	endbr64 
    1414:	f2 ff 25 75 cb 00 00 	bnd jmpq *0xcb75(%rip)        # df90 <srand48@GLIBC_2.2.5>
    141b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001420 <__printf_chk@plt>:
    1420:	f3 0f 1e fa          	endbr64 
    1424:	f2 ff 25 6d cb 00 00 	bnd jmpq *0xcb6d(%rip)        # df98 <__printf_chk@GLIBC_2.3.4>
    142b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001430 <fopen@plt>:
    1430:	f3 0f 1e fa          	endbr64 
    1434:	f2 ff 25 65 cb 00 00 	bnd jmpq *0xcb65(%rip)        # dfa0 <fopen@GLIBC_2.2.5>
    143b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001440 <perror@plt>:
    1440:	f3 0f 1e fa          	endbr64 
    1444:	f2 ff 25 5d cb 00 00 	bnd jmpq *0xcb5d(%rip)        # dfa8 <perror@GLIBC_2.2.5>
    144b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001450 <exit@plt>:
    1450:	f3 0f 1e fa          	endbr64 
    1454:	f2 ff 25 55 cb 00 00 	bnd jmpq *0xcb55(%rip)        # dfb0 <exit@GLIBC_2.2.5>
    145b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001460 <fwrite@plt>:
    1460:	f3 0f 1e fa          	endbr64 
    1464:	f2 ff 25 4d cb 00 00 	bnd jmpq *0xcb4d(%rip)        # dfb8 <fwrite@GLIBC_2.2.5>
    146b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001470 <__fprintf_chk@plt>:
    1470:	f3 0f 1e fa          	endbr64 
    1474:	f2 ff 25 45 cb 00 00 	bnd jmpq *0xcb45(%rip)        # dfc0 <__fprintf_chk@GLIBC_2.3.4>
    147b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001480 <strdup@plt>:
    1480:	f3 0f 1e fa          	endbr64 
    1484:	f2 ff 25 3d cb 00 00 	bnd jmpq *0xcb3d(%rip)        # dfc8 <strdup@GLIBC_2.2.5>
    148b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001490 <__sprintf_chk@plt>:
    1490:	f3 0f 1e fa          	endbr64 
    1494:	f2 ff 25 35 cb 00 00 	bnd jmpq *0xcb35(%rip)        # dfd0 <__sprintf_chk@GLIBC_2.3.4>
    149b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .text:

00000000000014a0 <main>:
    14a0:	f3 0f 1e fa          	endbr64 
    14a4:	41 57                	push   %r15
    14a6:	41 56                	push   %r14
    14a8:	41 55                	push   %r13
    14aa:	41 54                	push   %r12
    14ac:	41 89 fc             	mov    %edi,%r12d
    14af:	bf 03 00 00 00       	mov    $0x3,%edi
    14b4:	55                   	push   %rbp
    14b5:	53                   	push   %rbx
    14b6:	48 89 f3             	mov    %rsi,%rbx
    14b9:	48 8d 35 50 13 00 00 	lea    0x1350(%rip),%rsi        # 2810 <ctrl_bs_handler>
    14c0:	48 81 ec 98 01 00 00 	sub    $0x198,%rsp
    14c7:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    14ce:	00 00 
    14d0:	48 89 84 24 88 01 00 	mov    %rax,0x188(%rsp)
    14d7:	00 
    14d8:	31 c0                	xor    %eax,%eax
    14da:	48 8d ac 24 b0 00 00 	lea    0xb0(%rsp),%rbp
    14e1:	00 
    14e2:	e8 a9 fe ff ff       	callq  1390 <signal@plt>
    14e7:	31 ff                	xor    %edi,%edi
    14e9:	48 89 ee             	mov    %rbp,%rsi
    14ec:	e8 df fd ff ff       	callq  12d0 <clock_gettime@plt>
    14f1:	41 83 fc 08          	cmp    $0x8,%r12d
    14f5:	0f 84 98 00 00 00    	je     1593 <main+0xf3>
    14fb:	48 8b 0d 3e fc 00 00 	mov    0xfc3e(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    1502:	ba 3e 00 00 00       	mov    $0x3e,%edx
    1507:	be 01 00 00 00       	mov    $0x1,%esi
    150c:	48 8d 3d 85 8d 00 00 	lea    0x8d85(%rip),%rdi        # a298 <_IO_stdin_used+0x298>
    1513:	e8 48 ff ff ff       	callq  1460 <fwrite@plt>
    1518:	ba 91 00 00 00       	mov    $0x91,%edx
    151d:	48 8b 0d 1c fc 00 00 	mov    0xfc1c(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    1524:	be 01 00 00 00       	mov    $0x1,%esi
    1529:	48 8d 3d a8 8d 00 00 	lea    0x8da8(%rip),%rdi        # a2d8 <_IO_stdin_used+0x2d8>
    1530:	e8 2b ff ff ff       	callq  1460 <fwrite@plt>
    1535:	ba 2a 00 00 00       	mov    $0x2a,%edx
    153a:	48 8b 0d ff fb 00 00 	mov    0xfbff(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    1541:	be 01 00 00 00       	mov    $0x1,%esi
    1546:	48 8d 3d 23 8e 00 00 	lea    0x8e23(%rip),%rdi        # a370 <_IO_stdin_used+0x370>
    154d:	e8 0e ff ff ff       	callq  1460 <fwrite@plt>
    1552:	ba 3f 00 00 00       	mov    $0x3f,%edx
    1557:	48 8b 0d e2 fb 00 00 	mov    0xfbe2(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    155e:	be 01 00 00 00       	mov    $0x1,%esi
    1563:	48 8d 3d 36 8e 00 00 	lea    0x8e36(%rip),%rdi        # a3a0 <_IO_stdin_used+0x3a0>
    156a:	e8 f1 fe ff ff       	callq  1460 <fwrite@plt>
    156f:	ba 7c 00 00 00       	mov    $0x7c,%edx
    1574:	be 01 00 00 00       	mov    $0x1,%esi
    1579:	48 8b 0d c0 fb 00 00 	mov    0xfbc0(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    1580:	48 8d 3d 59 8e 00 00 	lea    0x8e59(%rip),%rdi        # a3e0 <_IO_stdin_used+0x3e0>
    1587:	e8 d4 fe ff ff       	callq  1460 <fwrite@plt>
    158c:	31 ff                	xor    %edi,%edi
    158e:	e8 bd fe ff ff       	callq  1450 <exit@plt>
    1593:	48 8b 7b 10          	mov    0x10(%rbx),%rdi
    1597:	31 c0                	xor    %eax,%eax
    1599:	48 8d 94 24 90 00 00 	lea    0x90(%rsp),%rdx
    15a0:	00 
    15a1:	48 8d 35 8f 8a 00 00 	lea    0x8a8f(%rip),%rsi        # a037 <_IO_stdin_used+0x37>
    15a8:	e8 53 fe ff ff       	callq  1400 <__isoc99_sscanf@plt>
    15ad:	85 c0                	test   %eax,%eax
    15af:	74 0a                	je     15bb <main+0x11b>
    15b1:	83 bc 24 90 00 00 00 	cmpl   $0x0,0x90(%rsp)
    15b8:	00 
    15b9:	7f 24                	jg     15df <main+0x13f>
    15bb:	48 8b 0d 7e fb 00 00 	mov    0xfb7e(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    15c2:	ba 14 00 00 00       	mov    $0x14,%edx
    15c7:	be 01 00 00 00       	mov    $0x1,%esi
    15cc:	48 8d 3d 67 8a 00 00 	lea    0x8a67(%rip),%rdi        # a03a <_IO_stdin_used+0x3a>
    15d3:	e8 88 fe ff ff       	callq  1460 <fwrite@plt>
    15d8:	31 ff                	xor    %edi,%edi
    15da:	e8 71 fe ff ff       	callq  1450 <exit@plt>
    15df:	48 8b 7b 18          	mov    0x18(%rbx),%rdi
    15e3:	31 c0                	xor    %eax,%eax
    15e5:	48 8d 94 24 94 00 00 	lea    0x94(%rsp),%rdx
    15ec:	00 
    15ed:	48 8d 35 43 8a 00 00 	lea    0x8a43(%rip),%rsi        # a037 <_IO_stdin_used+0x37>
    15f4:	e8 07 fe ff ff       	callq  1400 <__isoc99_sscanf@plt>
    15f9:	85 c0                	test   %eax,%eax
    15fb:	74 0a                	je     1607 <main+0x167>
    15fd:	83 bc 24 94 00 00 00 	cmpl   $0x0,0x94(%rsp)
    1604:	00 
    1605:	79 24                	jns    162b <main+0x18b>
    1607:	48 8b 0d 32 fb 00 00 	mov    0xfb32(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    160e:	ba 14 00 00 00       	mov    $0x14,%edx
    1613:	be 01 00 00 00       	mov    $0x1,%esi
    1618:	48 8d 3d 30 8a 00 00 	lea    0x8a30(%rip),%rdi        # a04f <_IO_stdin_used+0x4f>
    161f:	e8 3c fe ff ff       	callq  1460 <fwrite@plt>
    1624:	31 ff                	xor    %edi,%edi
    1626:	e8 25 fe ff ff       	callq  1450 <exit@plt>
    162b:	48 8b 7b 28          	mov    0x28(%rbx),%rdi
    162f:	31 c0                	xor    %eax,%eax
    1631:	48 8d 94 24 98 00 00 	lea    0x98(%rsp),%rdx
    1638:	00 
    1639:	48 8d 35 f7 89 00 00 	lea    0x89f7(%rip),%rsi        # a037 <_IO_stdin_used+0x37>
    1640:	e8 bb fd ff ff       	callq  1400 <__isoc99_sscanf@plt>
    1645:	85 c0                	test   %eax,%eax
    1647:	74 0a                	je     1653 <main+0x1b3>
    1649:	83 bc 24 98 00 00 00 	cmpl   $0x0,0x98(%rsp)
    1650:	00 
    1651:	7f 24                	jg     1677 <main+0x1d7>
    1653:	48 8b 0d e6 fa 00 00 	mov    0xfae6(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    165a:	ba 15 00 00 00       	mov    $0x15,%edx
    165f:	be 01 00 00 00       	mov    $0x1,%esi
    1664:	48 8d 3d f9 89 00 00 	lea    0x89f9(%rip),%rdi        # a064 <_IO_stdin_used+0x64>
    166b:	e8 f0 fd ff ff       	callq  1460 <fwrite@plt>
    1670:	31 ff                	xor    %edi,%edi
    1672:	e8 d9 fd ff ff       	callq  1450 <exit@plt>
    1677:	48 8b 7b 30          	mov    0x30(%rbx),%rdi
    167b:	31 c0                	xor    %eax,%eax
    167d:	48 8d 94 24 9c 00 00 	lea    0x9c(%rsp),%rdx
    1684:	00 
    1685:	48 8d 35 ab 89 00 00 	lea    0x89ab(%rip),%rsi        # a037 <_IO_stdin_used+0x37>
    168c:	e8 6f fd ff ff       	callq  1400 <__isoc99_sscanf@plt>
    1691:	85 c0                	test   %eax,%eax
    1693:	74 0a                	je     169f <main+0x1ff>
    1695:	83 bc 24 9c 00 00 00 	cmpl   $0x0,0x9c(%rsp)
    169c:	00 
    169d:	79 24                	jns    16c3 <main+0x223>
    169f:	48 8b 0d 9a fa 00 00 	mov    0xfa9a(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    16a6:	ba 1e 00 00 00       	mov    $0x1e,%edx
    16ab:	be 01 00 00 00       	mov    $0x1,%esi
    16b0:	48 8d 3d a9 8d 00 00 	lea    0x8da9(%rip),%rdi        # a460 <_IO_stdin_used+0x460>
    16b7:	e8 a4 fd ff ff       	callq  1460 <fwrite@plt>
    16bc:	31 ff                	xor    %edi,%edi
    16be:	e8 8d fd ff ff       	callq  1450 <exit@plt>
    16c3:	48 8b 7b 38          	mov    0x38(%rbx),%rdi
    16c7:	31 c0                	xor    %eax,%eax
    16c9:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
    16d0:	00 
    16d1:	48 8d 35 a2 89 00 00 	lea    0x89a2(%rip),%rsi        # a07a <_IO_stdin_used+0x7a>
    16d8:	e8 23 fd ff ff       	callq  1400 <__isoc99_sscanf@plt>
    16dd:	85 c0                	test   %eax,%eax
    16df:	0f 84 ae 01 00 00    	je     1893 <main+0x3f3>
    16e5:	48 8b 8c 24 a8 00 00 	mov    0xa8(%rsp),%rcx
    16ec:	00 
    16ed:	48 85 c9             	test   %rcx,%rcx
    16f0:	0f 88 9d 01 00 00    	js     1893 <main+0x3f3>
    16f6:	74 66                	je     175e <main+0x2be>
    16f8:	48 63 b4 24 98 00 00 	movslq 0x98(%rsp),%rsi
    16ff:	00 
    1700:	48 63 94 24 90 00 00 	movslq 0x90(%rsp),%rdx
    1707:	00 
    1708:	48 63 84 24 9c 00 00 	movslq 0x9c(%rsp),%rax
    170f:	00 
    1710:	48 0f af f2          	imul   %rdx,%rsi
    1714:	85 c0                	test   %eax,%eax
    1716:	0f 85 28 05 00 00    	jne    1c44 <main+0x7a4>
    171c:	48 c1 e6 03          	shl    $0x3,%rsi
    1720:	48 89 c8             	mov    %rcx,%rax
    1723:	31 d2                	xor    %edx,%edx
    1725:	48 f7 f6             	div    %rsi
    1728:	48 85 d2             	test   %rdx,%rdx
    172b:	74 31                	je     175e <main+0x2be>
    172d:	48 89 cf             	mov    %rcx,%rdi
    1730:	48 89 f8             	mov    %rdi,%rax
    1733:	31 d2                	xor    %edx,%edx
    1735:	48 f7 f6             	div    %rsi
    1738:	48 85 d2             	test   %rdx,%rdx
    173b:	0f 84 aa 0f 00 00    	je     26eb <main+0x124b>
    1741:	48 83 c1 01          	add    $0x1,%rcx
    1745:	31 d2                	xor    %edx,%edx
    1747:	48 83 ef 01          	sub    $0x1,%rdi
    174b:	48 89 c8             	mov    %rcx,%rax
    174e:	48 f7 f6             	div    %rsi
    1751:	48 85 d2             	test   %rdx,%rdx
    1754:	75 da                	jne    1730 <main+0x290>
    1756:	48 89 8c 24 a8 00 00 	mov    %rcx,0xa8(%rsp)
    175d:	00 
    175e:	4c 8b 63 20          	mov    0x20(%rbx),%r12
    1762:	48 8d 35 34 89 00 00 	lea    0x8934(%rip),%rsi        # a09d <_IO_stdin_used+0x9d>
    1769:	4c 89 e7             	mov    %r12,%rdi
    176c:	e8 0f fc ff ff       	callq  1380 <strcmp@plt>
    1771:	85 c0                	test   %eax,%eax
    1773:	0f 84 87 01 00 00    	je     1900 <main+0x460>
    1779:	48 8d 35 27 89 00 00 	lea    0x8927(%rip),%rsi        # a0a7 <_IO_stdin_used+0xa7>
    1780:	4c 89 e7             	mov    %r12,%rdi
    1783:	e8 f8 fb ff ff       	callq  1380 <strcmp@plt>
    1788:	85 c0                	test   %eax,%eax
    178a:	0f 85 27 01 00 00    	jne    18b7 <main+0x417>
    1790:	8b 84 24 98 00 00 00 	mov    0x98(%rsp),%eax
    1797:	8d 50 f8             	lea    -0x8(%rax),%edx
    179a:	83 e2 f7             	and    $0xfffffff7,%edx
    179d:	74 09                	je     17a8 <main+0x308>
    179f:	83 f8 20             	cmp    $0x20,%eax
    17a2:	0f 85 f6 04 00 00    	jne    1c9e <main+0x7fe>
    17a8:	c7 44 24 68 00 00 00 	movl   $0x0,0x68(%rsp)
    17af:	00 
    17b0:	8b 44 24 68          	mov    0x68(%rsp),%eax
    17b4:	bf d4 03 00 00       	mov    $0x3d4,%edi
    17b9:	89 05 59 11 01 00    	mov    %eax,0x11159(%rip)        # 12918 <method>
    17bf:	e8 2c fc ff ff       	callq  13f0 <malloc@plt>
    17c4:	be d4 03 00 00       	mov    $0x3d4,%esi
    17c9:	48 89 c7             	mov    %rax,%rdi
    17cc:	49 89 c4             	mov    %rax,%r12
    17cf:	e8 7c fb ff ff       	callq  1350 <getcwd@plt>
    17d4:	49 39 c4             	cmp    %rax,%r12
    17d7:	0f 85 87 0c 00 00    	jne    2464 <main+0xfc4>
    17dd:	bf e8 03 00 00       	mov    $0x3e8,%edi
    17e2:	e8 09 fc ff ff       	callq  13f0 <malloc@plt>
    17e7:	ba e8 03 00 00       	mov    $0x3e8,%edx
    17ec:	4c 89 e6             	mov    %r12,%rsi
    17ef:	49 89 c7             	mov    %rax,%r15
    17f2:	48 89 c7             	mov    %rax,%rdi
    17f5:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    17fa:	e8 d1 fb ff ff       	callq  13d0 <__stpcpy_chk@plt>
    17ff:	4c 89 f9             	mov    %r15,%rcx
    1802:	ba 0c 00 00 00       	mov    $0xc,%edx
    1807:	48 8d 35 75 89 00 00 	lea    0x8975(%rip),%rsi        # a183 <_IO_stdin_used+0x183>
    180e:	48 29 c1             	sub    %rax,%rcx
    1811:	48 89 c7             	mov    %rax,%rdi
    1814:	48 81 c1 e8 03 00 00 	add    $0x3e8,%rcx
    181b:	e8 80 fb ff ff       	callq  13a0 <__memcpy_chk@plt>
    1820:	48 8b 7b 08          	mov    0x8(%rbx),%rdi
    1824:	80 3f 2d             	cmpb   $0x2d,(%rdi)
    1827:	0f 84 c8 03 00 00    	je     1bf5 <main+0x755>
    182d:	48 8d 35 5b 89 00 00 	lea    0x895b(%rip),%rsi        # a18f <_IO_stdin_used+0x18f>
    1834:	e8 f7 fb ff ff       	callq  1430 <fopen@plt>
    1839:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    183e:	48 85 c0             	test   %rax,%rax
    1841:	0f 84 a6 0a 00 00    	je     22ed <main+0xe4d>
    1847:	be c0 01 00 00       	mov    $0x1c0,%esi
    184c:	48 8d 3d 55 89 00 00 	lea    0x8955(%rip),%rdi        # a1a8 <_IO_stdin_used+0x1a8>
    1853:	e8 58 fa ff ff       	callq  12b0 <mkdir@plt>
    1858:	83 c0 01             	add    $0x1,%eax
    185b:	0f 85 ac 00 00 00    	jne    190d <main+0x46d>
    1861:	e8 2a fa ff ff       	callq  1290 <__errno_location@plt>
    1866:	83 38 11             	cmpl   $0x11,(%rax)
    1869:	0f 84 9e 00 00 00    	je     190d <main+0x46d>
    186f:	48 8b 0d ca f8 00 00 	mov    0xf8ca(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    1876:	ba 23 00 00 00       	mov    $0x23,%edx
    187b:	be 01 00 00 00       	mov    $0x1,%esi
    1880:	48 8d 3d 89 8d 00 00 	lea    0x8d89(%rip),%rdi        # a610 <_IO_stdin_used+0x610>
    1887:	e8 d4 fb ff ff       	callq  1460 <fwrite@plt>
    188c:	31 ff                	xor    %edi,%edi
    188e:	e8 bd fb ff ff       	callq  1450 <exit@plt>
    1893:	48 8b 0d a6 f8 00 00 	mov    0xf8a6(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    189a:	ba 1d 00 00 00       	mov    $0x1d,%edx
    189f:	be 01 00 00 00       	mov    $0x1,%esi
    18a4:	48 8d 3d d4 87 00 00 	lea    0x87d4(%rip),%rdi        # a07f <_IO_stdin_used+0x7f>
    18ab:	e8 b0 fb ff ff       	callq  1460 <fwrite@plt>
    18b0:	31 ff                	xor    %edi,%edi
    18b2:	e8 99 fb ff ff       	callq  1450 <exit@plt>
    18b7:	48 8d 35 14 88 00 00 	lea    0x8814(%rip),%rsi        # a0d2 <_IO_stdin_used+0xd2>
    18be:	4c 89 e7             	mov    %r12,%rdi
    18c1:	e8 ba fa ff ff       	callq  1380 <strcmp@plt>
    18c6:	85 c0                	test   %eax,%eax
    18c8:	0f 85 68 0a 00 00    	jne    2336 <main+0xe96>
    18ce:	83 bc 24 94 00 00 00 	cmpl   $0x2,0x94(%rsp)
    18d5:	02 
    18d6:	0f 84 35 0a 00 00    	je     2311 <main+0xe71>
    18dc:	48 8b 0d 5d f8 00 00 	mov    0xf85d(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    18e3:	ba 15 00 00 00       	mov    $0x15,%edx
    18e8:	be 01 00 00 00       	mov    $0x1,%esi
    18ed:	48 8d 3d ed 87 00 00 	lea    0x87ed(%rip),%rdi        # a0e1 <_IO_stdin_used+0xe1>
    18f4:	e8 67 fb ff ff       	callq  1460 <fwrite@plt>
    18f9:	31 ff                	xor    %edi,%edi
    18fb:	e8 50 fb ff ff       	callq  1450 <exit@plt>
    1900:	c7 44 24 68 09 00 00 	movl   $0x9,0x68(%rsp)
    1907:	00 
    1908:	e9 a3 fe ff ff       	jmpq   17b0 <main+0x310>
    190d:	48 8b 73 08          	mov    0x8(%rbx),%rsi
    1911:	48 8d 94 24 f0 00 00 	lea    0xf0(%rsp),%rdx
    1918:	00 
    1919:	bf 01 00 00 00       	mov    $0x1,%edi
    191e:	e8 bd fa ff ff       	callq  13e0 <__xstat@plt>
    1923:	48 8b 84 24 20 01 00 	mov    0x120(%rsp),%rax
    192a:	00 
    192b:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
    1932:	00 
    1933:	4c 63 84 24 90 00 00 	movslq 0x90(%rsp),%r8
    193a:	00 
    193b:	8b 8c 24 98 00 00 00 	mov    0x98(%rsp),%ecx
    1942:	8b 84 24 9c 00 00 00 	mov    0x9c(%rsp),%eax
    1949:	48 8b b4 24 a0 00 00 	mov    0xa0(%rsp),%rsi
    1950:	00 
    1951:	41 0f af c8          	imul   %r8d,%ecx
    1955:	85 c0                	test   %eax,%eax
    1957:	0f 84 67 02 00 00    	je     1bc4 <main+0x724>
    195d:	0f af c8             	imul   %eax,%ecx
    1960:	31 d2                	xor    %edx,%edx
    1962:	48 89 f0             	mov    %rsi,%rax
    1965:	48 89 f7             	mov    %rsi,%rdi
    1968:	48 63 c9             	movslq %ecx,%rcx
    196b:	48 c1 e1 03          	shl    $0x3,%rcx
    196f:	48 f7 f1             	div    %rcx
    1972:	48 85 d2             	test   %rdx,%rdx
    1975:	0f 85 14 02 00 00    	jne    1b8f <main+0x6ef>
    197b:	4c 8b a4 24 a8 00 00 	mov    0xa8(%rsp),%r12
    1982:	00 
    1983:	4d 85 e4             	test   %r12,%r12
    1986:	0f 85 c0 01 00 00    	jne    1b4c <main+0x6ac>
    198c:	48 89 f8             	mov    %rdi,%rax
    198f:	48 89 b4 24 a8 00 00 	mov    %rsi,0xa8(%rsp)
    1996:	00 
    1997:	c7 05 6f 0f 01 00 01 	movl   $0x1,0x10f6f(%rip)        # 12910 <readins>
    199e:	00 00 00 
    19a1:	48 99                	cqto   
    19a3:	49 f7 f8             	idiv   %r8
    19a6:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    19ab:	e8 40 fa ff ff       	callq  13f0 <malloc@plt>
    19b0:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    19b5:	4c 8b 63 08          	mov    0x8(%rbx),%r12
    19b9:	31 c0                	xor    %eax,%eax
    19bb:	48 83 c9 ff          	or     $0xffffffffffffffff,%rcx
    19bf:	4c 89 e7             	mov    %r12,%rdi
    19c2:	f2 ae                	repnz scas %es:(%rdi),%al
    19c4:	48 f7 d1             	not    %rcx
    19c7:	48 8d 79 13          	lea    0x13(%rcx),%rdi
    19cb:	49 89 cd             	mov    %rcx,%r13
    19ce:	e8 1d fa ff ff       	callq  13f0 <malloc@plt>
    19d3:	be 2f 00 00 00       	mov    $0x2f,%esi
    19d8:	4c 89 e7             	mov    %r12,%rdi
    19db:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    19e0:	49 89 c7             	mov    %rax,%r15
    19e3:	e8 38 f9 ff ff       	callq  1320 <strrchr@plt>
    19e8:	48 85 c0             	test   %rax,%rax
    19eb:	0f 84 98 02 00 00    	je     1c89 <main+0x7e9>
    19f1:	48 8d 70 01          	lea    0x1(%rax),%rsi
    19f5:	4c 89 ff             	mov    %r15,%rdi
    19f8:	e8 a3 f8 ff ff       	callq  12a0 <strcpy@plt>
    19fd:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
    1a02:	be 2e 00 00 00       	mov    $0x2e,%esi
    1a07:	e8 04 f9 ff ff       	callq  1310 <strchr@plt>
    1a0c:	49 89 c5             	mov    %rax,%r13
    1a0f:	48 85 c0             	test   %rax,%rax
    1a12:	0f 84 5b 02 00 00    	je     1c73 <main+0x7d3>
    1a18:	48 89 c7             	mov    %rax,%rdi
    1a1b:	e8 60 fa ff ff       	callq  1480 <strdup@plt>
    1a20:	41 c6 45 00 00       	movb   $0x0,0x0(%r13)
    1a25:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    1a2a:	49 83 c9 ff          	or     $0xffffffffffffffff,%r9
    1a2e:	45 31 f6             	xor    %r14d,%r14d
    1a31:	4c 89 e7             	mov    %r12,%rdi
    1a34:	4c 89 c9             	mov    %r9,%rcx
    1a37:	44 89 f0             	mov    %r14d,%eax
    1a3a:	f2 ae                	repnz scas %es:(%rdi),%al
    1a3c:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    1a41:	49 89 cc             	mov    %rcx,%r12
    1a44:	4c 89 c9             	mov    %r9,%rcx
    1a47:	f2 ae                	repnz scas %es:(%rdi),%al
    1a49:	49 f7 d4             	not    %r12
    1a4c:	48 f7 d1             	not    %rcx
    1a4f:	49 8d 7c 0c 12       	lea    0x12(%r12,%rcx,1),%rdi
    1a54:	4c 8d a4 24 83 01 00 	lea    0x183(%rsp),%r12
    1a5b:	00 
    1a5c:	e8 8f f9 ff ff       	callq  13f0 <malloc@plt>
    1a61:	44 8b 84 24 90 00 00 	mov    0x90(%rsp),%r8d
    1a68:	00 
    1a69:	ba 05 00 00 00       	mov    $0x5,%edx
    1a6e:	4c 89 e7             	mov    %r12,%rdi
    1a71:	be 01 00 00 00       	mov    $0x1,%esi
    1a76:	48 8d 0d ba 85 00 00 	lea    0x85ba(%rip),%rcx        # a037 <_IO_stdin_used+0x37>
    1a7d:	49 89 c5             	mov    %rax,%r13
    1a80:	31 c0                	xor    %eax,%eax
    1a82:	e8 09 fa ff ff       	callq  1490 <__sprintf_chk@plt>
    1a87:	49 83 c9 ff          	or     $0xffffffffffffffff,%r9
    1a8b:	4c 89 e7             	mov    %r12,%rdi
    1a8e:	44 89 f0             	mov    %r14d,%eax
    1a91:	4c 89 c9             	mov    %r9,%rcx
    1a94:	45 31 e4             	xor    %r12d,%r12d
    1a97:	f2 ae                	repnz scas %es:(%rdi),%al
    1a99:	48 63 bc 24 90 00 00 	movslq 0x90(%rsp),%rdi
    1aa0:	00 
    1aa1:	48 c1 e7 03          	shl    $0x3,%rdi
    1aa5:	48 f7 d1             	not    %rcx
    1aa8:	8d 41 ff             	lea    -0x1(%rcx),%eax
    1aab:	89 44 24 24          	mov    %eax,0x24(%rsp)
    1aaf:	e8 3c f9 ff ff       	callq  13f0 <malloc@plt>
    1ab4:	48 63 bc 24 94 00 00 	movslq 0x94(%rsp),%rdi
    1abb:	00 
    1abc:	49 89 c6             	mov    %rax,%r14
    1abf:	48 c1 e7 03          	shl    $0x3,%rdi
    1ac3:	e8 28 f9 ff ff       	callq  13f0 <malloc@plt>
    1ac8:	4c 63 7c 24 58       	movslq 0x58(%rsp),%r15
    1acd:	48 89 5c 24 28       	mov    %rbx,0x28(%rsp)
    1ad2:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    1ad7:	48 89 c3             	mov    %rax,%rbx
    1ada:	4c 89 f8             	mov    %r15,%rax
    1add:	4d 89 e7             	mov    %r12,%r15
    1ae0:	49 89 ec             	mov    %rbp,%r12
    1ae3:	48 89 c5             	mov    %rax,%rbp
    1ae6:	eb 19                	jmp    1b01 <main+0x661>
    1ae8:	48 89 ef             	mov    %rbp,%rdi
    1aeb:	e8 00 f9 ff ff       	callq  13f0 <malloc@plt>
    1af0:	4a 89 04 fb          	mov    %rax,(%rbx,%r15,8)
    1af4:	49 83 c7 01          	add    $0x1,%r15
    1af8:	48 85 c0             	test   %rax,%rax
    1afb:	0f 84 cf 0b 00 00    	je     26d0 <main+0x1230>
    1b01:	44 39 bc 24 94 00 00 	cmp    %r15d,0x94(%rsp)
    1b08:	00 
    1b09:	7f dd                	jg     1ae8 <main+0x648>
    1b0b:	48 8d 84 24 d0 00 00 	lea    0xd0(%rsp),%rax
    1b12:	00 
    1b13:	31 ff                	xor    %edi,%edi
    1b15:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
    1b1a:	4c 89 e5             	mov    %r12,%rbp
    1b1d:	48 89 c6             	mov    %rax,%rsi
    1b20:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    1b25:	e8 a6 f7 ff ff       	callq  12d0 <clock_gettime@plt>
    1b2a:	8b 44 24 68          	mov    0x68(%rsp),%eax
    1b2e:	83 f8 08             	cmp    $0x8,%eax
    1b31:	0f 87 4c 09 00 00    	ja     2483 <main+0xfe3>
    1b37:	48 8d 15 42 8b 00 00 	lea    0x8b42(%rip),%rdx        # a680 <_IO_stdin_used+0x680>
    1b3e:	48 63 04 82          	movslq (%rdx,%rax,4),%rax
    1b42:	48 01 d0             	add    %rdx,%rax
    1b45:	3e ff e0             	notrack jmpq *%rax
    1b48:	48 83 c7 01          	add    $0x1,%rdi
    1b4c:	48 89 f8             	mov    %rdi,%rax
    1b4f:	48 99                	cqto   
    1b51:	49 f7 fc             	idiv   %r12
    1b54:	48 85 d2             	test   %rdx,%rdx
    1b57:	75 ef                	jne    1b48 <main+0x6a8>
    1b59:	49 39 f4             	cmp    %rsi,%r12
    1b5c:	0f 8d 2a fe ff ff    	jge    198c <main+0x4ec>
    1b62:	4c 89 e7             	mov    %r12,%rdi
    1b65:	89 05 a5 0d 01 00    	mov    %eax,0x10da5(%rip)        # 12910 <readins>
    1b6b:	e8 80 f8 ff ff       	callq  13f0 <malloc@plt>
    1b70:	48 63 8c 24 90 00 00 	movslq 0x90(%rsp),%rcx
    1b77:	00 
    1b78:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    1b7d:	4c 89 e0             	mov    %r12,%rax
    1b80:	48 99                	cqto   
    1b82:	48 f7 f9             	idiv   %rcx
    1b85:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    1b8a:	e9 26 fe ff ff       	jmpq   19b5 <main+0x515>
    1b8f:	48 83 c7 01          	add    $0x1,%rdi
    1b93:	31 d2                	xor    %edx,%edx
    1b95:	48 89 f8             	mov    %rdi,%rax
    1b98:	48 f7 f1             	div    %rcx
    1b9b:	48 85 d2             	test   %rdx,%rdx
    1b9e:	75 ef                	jne    1b8f <main+0x6ef>
    1ba0:	e9 d6 fd ff ff       	jmpq   197b <main+0x4db>
    1ba5:	48 8d 0d 25 8b 00 00 	lea    0x8b25(%rip),%rcx        # a6d1 <__PRETTY_FUNCTION__.5230>
    1bac:	ba d8 01 00 00       	mov    $0x1d8,%edx
    1bb1:	48 8d 35 73 84 00 00 	lea    0x8473(%rip),%rsi        # a02b <_IO_stdin_used+0x2b>
    1bb8:	48 8d 3d 76 84 00 00 	lea    0x8476(%rip),%rdi        # a035 <_IO_stdin_used+0x35>
    1bbf:	e8 6c f7 ff ff       	callq  1330 <__assert_fail@plt>
    1bc4:	48 63 c9             	movslq %ecx,%rcx
    1bc7:	48 89 f0             	mov    %rsi,%rax
    1bca:	31 d2                	xor    %edx,%edx
    1bcc:	48 89 f7             	mov    %rsi,%rdi
    1bcf:	48 c1 e1 03          	shl    $0x3,%rcx
    1bd3:	48 f7 f1             	div    %rcx
    1bd6:	48 85 d2             	test   %rdx,%rdx
    1bd9:	0f 84 9c fd ff ff    	je     197b <main+0x4db>
    1bdf:	48 83 c7 01          	add    $0x1,%rdi
    1be3:	31 d2                	xor    %edx,%edx
    1be5:	48 89 f8             	mov    %rdi,%rax
    1be8:	48 f7 f1             	div    %rcx
    1beb:	48 85 d2             	test   %rdx,%rdx
    1bee:	75 ef                	jne    1bdf <main+0x73f>
    1bf0:	e9 86 fd ff ff       	jmpq   197b <main+0x4db>
    1bf5:	48 83 c7 01          	add    $0x1,%rdi
    1bf9:	31 c0                	xor    %eax,%eax
    1bfb:	48 8d 94 24 a0 00 00 	lea    0xa0(%rsp),%rdx
    1c02:	00 
    1c03:	48 8d 35 70 84 00 00 	lea    0x8470(%rip),%rsi        # a07a <_IO_stdin_used+0x7a>
    1c0a:	e8 f1 f7 ff ff       	callq  1400 <__isoc99_sscanf@plt>
    1c0f:	83 e8 01             	sub    $0x1,%eax
    1c12:	0f 85 e0 0a 00 00    	jne    26f8 <main+0x1258>
    1c18:	48 83 bc 24 a0 00 00 	cmpq   $0x0,0xa0(%rsp)
    1c1f:	00 00 
    1c21:	0f 8e d1 0a 00 00    	jle    26f8 <main+0x1258>
    1c27:	31 ff                	xor    %edi,%edi
    1c29:	e8 92 f7 ff ff       	callq  13c0 <time@plt>
    1c2e:	48 89 c7             	mov    %rax,%rdi
    1c31:	e8 da f7 ff ff       	callq  1410 <srand48@plt>
    1c36:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    1c3d:	00 00 
    1c3f:	e9 ef fc ff ff       	jmpq   1933 <main+0x493>
    1c44:	48 0f af f0          	imul   %rax,%rsi
    1c48:	31 d2                	xor    %edx,%edx
    1c4a:	48 89 c8             	mov    %rcx,%rax
    1c4d:	48 c1 e6 03          	shl    $0x3,%rsi
    1c51:	48 f7 f6             	div    %rsi
    1c54:	48 85 d2             	test   %rdx,%rdx
    1c57:	0f 84 01 fb ff ff    	je     175e <main+0x2be>
    1c5d:	48 83 c1 01          	add    $0x1,%rcx
    1c61:	31 d2                	xor    %edx,%edx
    1c63:	48 89 c8             	mov    %rcx,%rax
    1c66:	48 f7 f6             	div    %rsi
    1c69:	48 85 d2             	test   %rdx,%rdx
    1c6c:	75 ef                	jne    1c5d <main+0x7bd>
    1c6e:	e9 e3 fa ff ff       	jmpq   1756 <main+0x2b6>
    1c73:	48 8d 3d b0 83 00 00 	lea    0x83b0(%rip),%rdi        # a02a <_IO_stdin_used+0x2a>
    1c7a:	e8 01 f8 ff ff       	callq  1480 <strdup@plt>
    1c7f:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    1c84:	e9 a1 fd ff ff       	jmpq   1a2a <main+0x58a>
    1c89:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
    1c8e:	4c 89 ea             	mov    %r13,%rdx
    1c91:	4c 89 e6             	mov    %r12,%rsi
    1c94:	e8 17 f7 ff ff       	callq  13b0 <memcpy@plt>
    1c99:	e9 5f fd ff ff       	jmpq   19fd <main+0x55d>
    1c9e:	48 8b 0d 9b f4 00 00 	mov    0xf49b(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    1ca5:	ba 1d 00 00 00       	mov    $0x1d,%edx
    1caa:	be 01 00 00 00       	mov    $0x1,%esi
    1caf:	48 8d 3d fe 83 00 00 	lea    0x83fe(%rip),%rdi        # a0b4 <_IO_stdin_used+0xb4>
    1cb6:	e8 a5 f7 ff ff       	callq  1460 <fwrite@plt>
    1cbb:	31 ff                	xor    %edi,%edi
    1cbd:	e8 8e f7 ff ff       	callq  1450 <exit@plt>
    1cc2:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    1cc9:	e8 72 10 00 00       	callq  2d40 <liber8tion_coding_bitmatrix>
    1cce:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    1cd5:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    1cdc:	48 89 c1             	mov    %rax,%rcx
    1cdf:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    1ce6:	e8 55 45 00 00       	callq  6240 <jerasure_smart_bitmatrix_to_schedule>
    1ceb:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
    1cf2:	00 00 
    1cf4:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    1cf9:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
    1d00:	00 
    1d01:	31 ff                	xor    %edi,%edi
    1d03:	48 89 c6             	mov    %rax,%rsi
    1d06:	49 89 c7             	mov    %rax,%r15
    1d09:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    1d0e:	e8 bd f5 ff ff       	callq  12d0 <clock_gettime@plt>
    1d13:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    1d18:	4c 89 fe             	mov    %r15,%rsi
    1d1b:	e8 30 7c 00 00       	callq  9950 <timing_delta>
    1d20:	f2 0f 58 05 b0 89 00 	addsd  0x89b0(%rip),%xmm0        # a6d8 <__PRETTY_FUNCTION__.5230+0x7>
    1d27:	00 
    1d28:	c7 05 e2 0b 01 00 01 	movl   $0x1,0x10be2(%rip)        # 12914 <n>
    1d2f:	00 00 00 
    1d32:	c7 44 24 6c 00 00 00 	movl   $0x0,0x6c(%rsp)
    1d39:	00 
    1d3a:	4c 63 64 24 58       	movslq 0x58(%rsp),%r12
    1d3f:	48 89 9c 24 80 00 00 	mov    %rbx,0x80(%rsp)
    1d46:	00 
    1d47:	4c 8b 7c 24 78       	mov    0x78(%rsp),%r15
    1d4c:	48 89 ac 24 88 00 00 	mov    %rbp,0x88(%rsp)
    1d53:	00 
    1d54:	f2 0f 11 44 24 30    	movsd  %xmm0,0x30(%rsp)
    1d5a:	8b 05 b0 0b 01 00    	mov    0x10bb0(%rip),%eax        # 12910 <readins>
    1d60:	39 05 ae 0b 01 00    	cmp    %eax,0x10bae(%rip)        # 12914 <n>
    1d66:	0f 8f 45 03 00 00    	jg     20b1 <main+0xc11>
    1d6c:	48 63 44 24 6c       	movslq 0x6c(%rsp),%rax
    1d71:	48 8b 94 24 a0 00 00 	mov    0xa0(%rsp),%rdx
    1d78:	00 
    1d79:	48 39 d0             	cmp    %rdx,%rax
    1d7c:	0f 8c ae 02 00 00    	jl     2030 <main+0xb90>
    1d82:	0f 84 06 03 00 00    	je     208e <main+0xbee>
    1d88:	8b 8c 24 90 00 00 00 	mov    0x90(%rsp),%ecx
    1d8f:	48 8b 54 24 48       	mov    0x48(%rsp),%rdx
    1d94:	31 c0                	xor    %eax,%eax
    1d96:	eb 0b                	jmp    1da3 <main+0x903>
    1d98:	49 89 14 c6          	mov    %rdx,(%r14,%rax,8)
    1d9c:	48 83 c0 01          	add    $0x1,%rax
    1da0:	4c 01 e2             	add    %r12,%rdx
    1da3:	39 c1                	cmp    %eax,%ecx
    1da5:	7f f1                	jg     1d98 <main+0x8f8>
    1da7:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
    1dac:	31 ff                	xor    %edi,%edi
    1dae:	e8 1d f5 ff ff       	callq  12d0 <clock_gettime@plt>
    1db3:	8b 44 24 68          	mov    0x68(%rsp),%eax
    1db7:	83 f8 08             	cmp    $0x8,%eax
    1dba:	77 48                	ja     1e04 <main+0x964>
    1dbc:	48 8d 1d e1 88 00 00 	lea    0x88e1(%rip),%rbx        # a6a4 <_IO_stdin_used+0x6a4>
    1dc3:	48 63 04 83          	movslq (%rbx,%rax,4),%rax
    1dc7:	48 01 d8             	add    %rbx,%rax
    1dca:	3e ff e0             	notrack jmpq *%rax
    1dcd:	8b 84 24 9c 00 00 00 	mov    0x9c(%rsp),%eax
    1dd4:	4d 89 f0             	mov    %r14,%r8
    1dd7:	50                   	push   %rax
    1dd8:	8b 44 24 60          	mov    0x60(%rsp),%eax
    1ddc:	50                   	push   %rax
    1ddd:	4c 8b 4c 24 48       	mov    0x48(%rsp),%r9
    1de2:	48 8b 4c 24 70       	mov    0x70(%rsp),%rcx
    1de7:	8b 94 24 a8 00 00 00 	mov    0xa8(%rsp),%edx
    1dee:	8b b4 24 a4 00 00 00 	mov    0xa4(%rsp),%esi
    1df5:	8b bc 24 a0 00 00 00 	mov    0xa0(%rsp),%edi
    1dfc:	e8 9f 41 00 00       	callq  5fa0 <jerasure_schedule_encode>
    1e01:	5f                   	pop    %rdi
    1e02:	41 58                	pop    %r8
    1e04:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
    1e09:	31 ff                	xor    %edi,%edi
    1e0b:	bb 01 00 00 00       	mov    $0x1,%ebx
    1e10:	e8 bb f4 ff ff       	callq  12d0 <clock_gettime@plt>
    1e15:	eb 71                	jmp    1e88 <main+0x9e8>
    1e17:	56                   	push   %rsi
    1e18:	48 83 ca ff          	or     $0xffffffffffffffff,%rdx
    1e1c:	4d 89 f9             	mov    %r15,%r9
    1e1f:	48 8d 0d 9b 83 00 00 	lea    0x839b(%rip),%rcx        # a1c1 <_IO_stdin_used+0x1c1>
    1e26:	ff 74 24 20          	pushq  0x20(%rsp)
    1e2a:	be 01 00 00 00       	mov    $0x1,%esi
    1e2f:	4c 89 ef             	mov    %r13,%rdi
    1e32:	53                   	push   %rbx
    1e33:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
    1e37:	50                   	push   %rax
    1e38:	4c 8b 44 24 30       	mov    0x30(%rsp),%r8
    1e3d:	31 c0                	xor    %eax,%eax
    1e3f:	e8 4c f6 ff ff       	callq  1490 <__sprintf_chk@plt>
    1e44:	48 83 c4 20          	add    $0x20,%rsp
    1e48:	83 3d c5 0a 01 00 01 	cmpl   $0x1,0x10ac5(%rip)        # 12914 <n>
    1e4f:	0f 84 da 00 00 00    	je     1f2f <main+0xa8f>
    1e55:	48 8d 35 7d 83 00 00 	lea    0x837d(%rip),%rsi        # a1d9 <_IO_stdin_used+0x1d9>
    1e5c:	4c 89 ef             	mov    %r13,%rdi
    1e5f:	e8 cc f5 ff ff       	callq  1430 <fopen@plt>
    1e64:	48 89 c5             	mov    %rax,%rbp
    1e67:	49 8b 7c de f8       	mov    -0x8(%r14,%rbx,8),%rdi
    1e6c:	48 89 e9             	mov    %rbp,%rcx
    1e6f:	4c 89 e2             	mov    %r12,%rdx
    1e72:	be 01 00 00 00       	mov    $0x1,%esi
    1e77:	e8 e4 f5 ff ff       	callq  1460 <fwrite@plt>
    1e7c:	48 89 ef             	mov    %rbp,%rdi
    1e7f:	e8 5c f4 ff ff       	callq  12e0 <fclose@plt>
    1e84:	48 83 c3 01          	add    $0x1,%rbx
    1e88:	39 9c 24 90 00 00 00 	cmp    %ebx,0x90(%rsp)
    1e8f:	0f 8c b1 00 00 00    	jl     1f46 <main+0xaa6>
    1e95:	48 83 7c 24 08 00    	cmpq   $0x0,0x8(%rsp)
    1e9b:	0f 85 76 ff ff ff    	jne    1e17 <main+0x977>
    1ea1:	49 8b 7c de f8       	mov    -0x8(%r14,%rbx,8),%rdi
    1ea6:	4c 89 e2             	mov    %r12,%rdx
    1ea9:	31 f6                	xor    %esi,%esi
    1eab:	e8 90 f4 ff ff       	callq  1340 <memset@plt>
    1eb0:	eb d2                	jmp    1e84 <main+0x9e4>
    1eb2:	48 8d 0d 18 88 00 00 	lea    0x8818(%rip),%rcx        # a6d1 <__PRETTY_FUNCTION__.5230>
    1eb9:	ba 14 02 00 00       	mov    $0x214,%edx
    1ebe:	48 8d 35 66 81 00 00 	lea    0x8166(%rip),%rsi        # a02b <_IO_stdin_used+0x2b>
    1ec5:	48 8d 3d 69 81 00 00 	lea    0x8169(%rip),%rdi        # a035 <_IO_stdin_used+0x35>
    1ecc:	e8 5f f4 ff ff       	callq  1330 <__assert_fail@plt>
    1ed1:	44 8b 44 24 58       	mov    0x58(%rsp),%r8d
    1ed6:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    1edb:	4c 89 f2             	mov    %r14,%rdx
    1ede:	8b b4 24 98 00 00 00 	mov    0x98(%rsp),%esi
    1ee5:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    1eec:	e8 bf 6a 00 00       	callq  89b0 <reed_sol_r6_encode>
    1ef1:	e9 0e ff ff ff       	jmpq   1e04 <main+0x964>
    1ef6:	41 51                	push   %r9
    1ef8:	8b 44 24 60          	mov    0x60(%rsp),%eax
    1efc:	4d 89 f0             	mov    %r14,%r8
    1eff:	50                   	push   %rax
    1f00:	4c 8b 4c 24 48       	mov    0x48(%rsp),%r9
    1f05:	48 8b 8c 24 80 00 00 	mov    0x80(%rsp),%rcx
    1f0c:	00 
    1f0d:	8b 94 24 a8 00 00 00 	mov    0xa8(%rsp),%edx
    1f14:	8b b4 24 a4 00 00 00 	mov    0xa4(%rsp),%esi
    1f1b:	8b bc 24 a0 00 00 00 	mov    0xa0(%rsp),%edi
    1f22:	e8 d9 2f 00 00       	callq  4f00 <jerasure_matrix_encode>
    1f27:	41 5b                	pop    %r11
    1f29:	5b                   	pop    %rbx
    1f2a:	e9 d5 fe ff ff       	jmpq   1e04 <main+0x964>
    1f2f:	48 8d 35 a0 82 00 00 	lea    0x82a0(%rip),%rsi        # a1d6 <_IO_stdin_used+0x1d6>
    1f36:	4c 89 ef             	mov    %r13,%rdi
    1f39:	e8 f2 f4 ff ff       	callq  1430 <fopen@plt>
    1f3e:	48 89 c5             	mov    %rax,%rbp
    1f41:	e9 21 ff ff ff       	jmpq   1e67 <main+0x9c7>
    1f46:	bb 01 00 00 00       	mov    $0x1,%ebx
    1f4b:	4c 89 74 24 28       	mov    %r14,0x28(%rsp)
    1f50:	49 89 de             	mov    %rbx,%r14
    1f53:	48 8b 5c 24 38       	mov    0x38(%rsp),%rbx
    1f58:	eb 6e                	jmp    1fc8 <main+0xb28>
    1f5a:	51                   	push   %rcx
    1f5b:	48 83 ca ff          	or     $0xffffffffffffffff,%rdx
    1f5f:	4d 89 f9             	mov    %r15,%r9
    1f62:	48 8d 0d 73 82 00 00 	lea    0x8273(%rip),%rcx        # a1dc <_IO_stdin_used+0x1dc>
    1f69:	ff 74 24 20          	pushq  0x20(%rsp)
    1f6d:	be 01 00 00 00       	mov    $0x1,%esi
    1f72:	4c 89 ef             	mov    %r13,%rdi
    1f75:	41 56                	push   %r14
    1f77:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
    1f7b:	50                   	push   %rax
    1f7c:	4c 8b 44 24 30       	mov    0x30(%rsp),%r8
    1f81:	31 c0                	xor    %eax,%eax
    1f83:	e8 08 f5 ff ff       	callq  1490 <__sprintf_chk@plt>
    1f88:	48 83 c4 20          	add    $0x20,%rsp
    1f8c:	83 3d 81 09 01 00 01 	cmpl   $0x1,0x10981(%rip)        # 12914 <n>
    1f93:	74 5b                	je     1ff0 <main+0xb50>
    1f95:	48 8d 35 3d 82 00 00 	lea    0x823d(%rip),%rsi        # a1d9 <_IO_stdin_used+0x1d9>
    1f9c:	4c 89 ef             	mov    %r13,%rdi
    1f9f:	e8 8c f4 ff ff       	callq  1430 <fopen@plt>
    1fa4:	48 89 c5             	mov    %rax,%rbp
    1fa7:	4a 8b 7c f3 f8       	mov    -0x8(%rbx,%r14,8),%rdi
    1fac:	48 89 e9             	mov    %rbp,%rcx
    1faf:	4c 89 e2             	mov    %r12,%rdx
    1fb2:	be 01 00 00 00       	mov    $0x1,%esi
    1fb7:	e8 a4 f4 ff ff       	callq  1460 <fwrite@plt>
    1fbc:	48 89 ef             	mov    %rbp,%rdi
    1fbf:	e8 1c f3 ff ff       	callq  12e0 <fclose@plt>
    1fc4:	49 83 c6 01          	add    $0x1,%r14
    1fc8:	44 39 b4 24 94 00 00 	cmp    %r14d,0x94(%rsp)
    1fcf:	00 
    1fd0:	7c 32                	jl     2004 <main+0xb64>
    1fd2:	48 83 7c 24 08 00    	cmpq   $0x0,0x8(%rsp)
    1fd8:	75 80                	jne    1f5a <main+0xaba>
    1fda:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    1fdf:	4c 89 e2             	mov    %r12,%rdx
    1fe2:	31 f6                	xor    %esi,%esi
    1fe4:	4a 8b 7c f0 f8       	mov    -0x8(%rax,%r14,8),%rdi
    1fe9:	e8 52 f3 ff ff       	callq  1340 <memset@plt>
    1fee:	eb d4                	jmp    1fc4 <main+0xb24>
    1ff0:	48 8d 35 df 81 00 00 	lea    0x81df(%rip),%rsi        # a1d6 <_IO_stdin_used+0x1d6>
    1ff7:	4c 89 ef             	mov    %r13,%rdi
    1ffa:	e8 31 f4 ff ff       	callq  1430 <fopen@plt>
    1fff:	48 89 c5             	mov    %rax,%rbp
    2002:	eb a3                	jmp    1fa7 <main+0xb07>
    2004:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
    2009:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    200e:	83 05 ff 08 01 00 01 	addl   $0x1,0x108ff(%rip)        # 12914 <n>
    2015:	4c 8b 74 24 28       	mov    0x28(%rsp),%r14
    201a:	e8 31 79 00 00       	callq  9950 <timing_delta>
    201f:	f2 0f 58 44 24 30    	addsd  0x30(%rsp),%xmm0
    2025:	f2 0f 11 44 24 30    	movsd  %xmm0,0x30(%rsp)
    202b:	e9 2a fd ff ff       	jmpq   1d5a <main+0x8ba>
    2030:	4c 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%r8
    2037:	00 
    2038:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    203d:	be 01 00 00 00       	mov    $0x1,%esi
    2042:	4c 01 c0             	add    %r8,%rax
    2045:	48 39 c2             	cmp    %rax,%rdx
    2048:	4c 89 c2             	mov    %r8,%rdx
    204b:	7d 2e                	jge    207b <main+0xbdb>
    204d:	48 8b 5c 24 48       	mov    0x48(%rsp),%rbx
    2052:	48 89 df             	mov    %rbx,%rdi
    2055:	e8 b6 08 00 00       	callq  2910 <jfread>
    205a:	48 8b 94 24 a8 00 00 	mov    0xa8(%rsp),%rdx
    2061:	00 
    2062:	48 89 d9             	mov    %rbx,%rcx
    2065:	48 98                	cltq   
    2067:	eb 08                	jmp    2071 <main+0xbd1>
    2069:	c6 04 01 30          	movb   $0x30,(%rcx,%rax,1)
    206d:	48 83 c0 01          	add    $0x1,%rax
    2071:	48 39 c2             	cmp    %rax,%rdx
    2074:	7f f3                	jg     2069 <main+0xbc9>
    2076:	e9 0d fd ff ff       	jmpq   1d88 <main+0x8e8>
    207b:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
    2080:	e8 8b 08 00 00       	callq  2910 <jfread>
    2085:	01 44 24 6c          	add    %eax,0x6c(%rsp)
    2089:	e9 fa fc ff ff       	jmpq   1d88 <main+0x8e8>
    208e:	48 8b 94 24 a8 00 00 	mov    0xa8(%rsp),%rdx
    2095:	00 
    2096:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    209b:	31 c0                	xor    %eax,%eax
    209d:	eb 08                	jmp    20a7 <main+0xc07>
    209f:	c6 04 01 30          	movb   $0x30,(%rcx,%rax,1)
    20a3:	48 83 c0 01          	add    $0x1,%rax
    20a7:	48 39 d0             	cmp    %rdx,%rax
    20aa:	7c f3                	jl     209f <main+0xbff>
    20ac:	e9 d7 fc ff ff       	jmpq   1d88 <main+0x8e8>
    20b1:	48 83 7c 24 08 00    	cmpq   $0x0,0x8(%rsp)
    20b7:	48 8b 9c 24 80 00 00 	mov    0x80(%rsp),%rbx
    20be:	00 
    20bf:	48 8b ac 24 88 00 00 	mov    0x88(%rsp),%rbp
    20c6:	00 
    20c7:	0f 84 04 01 00 00    	je     21d1 <main+0xd31>
    20cd:	4c 8b 4c 24 78       	mov    0x78(%rsp),%r9
    20d2:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
    20d7:	48 83 ca ff          	or     $0xffffffffffffffff,%rdx
    20db:	31 c0                	xor    %eax,%eax
    20dd:	48 8d 0d 0d 81 00 00 	lea    0x810d(%rip),%rcx        # a1f1 <_IO_stdin_used+0x1f1>
    20e4:	be 01 00 00 00       	mov    $0x1,%esi
    20e9:	4c 89 ef             	mov    %r13,%rdi
    20ec:	e8 9f f3 ff ff       	callq  1490 <__sprintf_chk@plt>
    20f1:	48 8d 35 de 80 00 00 	lea    0x80de(%rip),%rsi        # a1d6 <_IO_stdin_used+0x1d6>
    20f8:	4c 89 ef             	mov    %r13,%rdi
    20fb:	e8 30 f3 ff ff       	callq  1430 <fopen@plt>
    2100:	48 8b 4b 08          	mov    0x8(%rbx),%rcx
    2104:	be 01 00 00 00       	mov    $0x1,%esi
    2109:	48 8d 15 f5 7e 00 00 	lea    0x7ef5(%rip),%rdx        # a005 <_IO_stdin_used+0x5>
    2110:	49 89 c4             	mov    %rax,%r12
    2113:	48 89 c7             	mov    %rax,%rdi
    2116:	31 c0                	xor    %eax,%eax
    2118:	e8 53 f3 ff ff       	callq  1470 <__fprintf_chk@plt>
    211d:	be 01 00 00 00       	mov    $0x1,%esi
    2122:	4c 89 e7             	mov    %r12,%rdi
    2125:	31 c0                	xor    %eax,%eax
    2127:	48 8b 8c 24 a0 00 00 	mov    0xa0(%rsp),%rcx
    212e:	00 
    212f:	48 8d 15 dd 80 00 00 	lea    0x80dd(%rip),%rdx        # a213 <_IO_stdin_used+0x213>
    2136:	e8 35 f3 ff ff       	callq  1470 <__fprintf_chk@plt>
    213b:	ff b4 24 a8 00 00 00 	pushq  0xa8(%rsp)
    2142:	be 01 00 00 00       	mov    $0x1,%esi
    2147:	4c 89 e7             	mov    %r12,%rdi
    214a:	8b 84 24 a4 00 00 00 	mov    0xa4(%rsp),%eax
    2151:	48 8d 15 af 80 00 00 	lea    0x80af(%rip),%rdx        # a207 <_IO_stdin_used+0x207>
    2158:	50                   	push   %rax
    2159:	44 8b 8c 24 a8 00 00 	mov    0xa8(%rsp),%r9d
    2160:	00 
    2161:	31 c0                	xor    %eax,%eax
    2163:	44 8b 84 24 a4 00 00 	mov    0xa4(%rsp),%r8d
    216a:	00 
    216b:	8b 8c 24 a0 00 00 00 	mov    0xa0(%rsp),%ecx
    2172:	e8 f9 f2 ff ff       	callq  1470 <__fprintf_chk@plt>
    2177:	48 8b 4b 20          	mov    0x20(%rbx),%rcx
    217b:	be 01 00 00 00       	mov    $0x1,%esi
    2180:	31 c0                	xor    %eax,%eax
    2182:	48 8d 15 7c 7e 00 00 	lea    0x7e7c(%rip),%rdx        # a005 <_IO_stdin_used+0x5>
    2189:	4c 89 e7             	mov    %r12,%rdi
    218c:	e8 df f2 ff ff       	callq  1470 <__fprintf_chk@plt>
    2191:	8b 4c 24 78          	mov    0x78(%rsp),%ecx
    2195:	be 01 00 00 00       	mov    $0x1,%esi
    219a:	31 c0                	xor    %eax,%eax
    219c:	48 8d 15 77 7e 00 00 	lea    0x7e77(%rip),%rdx        # a01a <_IO_stdin_used+0x1a>
    21a3:	4c 89 e7             	mov    %r12,%rdi
    21a6:	e8 c5 f2 ff ff       	callq  1470 <__fprintf_chk@plt>
    21ab:	8b 0d 5f 07 01 00    	mov    0x1075f(%rip),%ecx        # 12910 <readins>
    21b1:	4c 89 e7             	mov    %r12,%rdi
    21b4:	31 c0                	xor    %eax,%eax
    21b6:	48 8d 15 5d 7e 00 00 	lea    0x7e5d(%rip),%rdx        # a01a <_IO_stdin_used+0x1a>
    21bd:	be 01 00 00 00       	mov    $0x1,%esi
    21c2:	e8 a9 f2 ff ff       	callq  1470 <__fprintf_chk@plt>
    21c7:	4c 89 e7             	mov    %r12,%rdi
    21ca:	e8 11 f1 ff ff       	callq  12e0 <fclose@plt>
    21cf:	58                   	pop    %rax
    21d0:	5a                   	pop    %rdx
    21d1:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
    21d6:	4c 8d a4 24 c0 00 00 	lea    0xc0(%rsp),%r12
    21dd:	00 
    21de:	e8 8d f0 ff ff       	callq  1270 <free@plt>
    21e3:	4c 89 ef             	mov    %r13,%rdi
    21e6:	e8 85 f0 ff ff       	callq  1270 <free@plt>
    21eb:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
    21f0:	e8 7b f0 ff ff       	callq  1270 <free@plt>
    21f5:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    21fa:	e8 71 f0 ff ff       	callq  1270 <free@plt>
    21ff:	31 ff                	xor    %edi,%edi
    2201:	4c 89 e6             	mov    %r12,%rsi
    2204:	e8 c7 f0 ff ff       	callq  12d0 <clock_gettime@plt>
    2209:	4c 89 e6             	mov    %r12,%rsi
    220c:	48 89 ef             	mov    %rbp,%rdi
    220f:	e8 3c 77 00 00       	callq  9950 <timing_delta>
    2214:	bf 01 00 00 00       	mov    $0x1,%edi
    2219:	b8 01 00 00 00       	mov    $0x1,%eax
    221e:	f2 0f 10 0d ba 84 00 	movsd  0x84ba(%rip),%xmm1        # a6e0 <__PRETTY_FUNCTION__.5230+0xf>
    2225:	00 
    2226:	f2 0f 11 44 24 08    	movsd  %xmm0,0x8(%rsp)
    222c:	66 0f ef c0          	pxor   %xmm0,%xmm0
    2230:	48 8d 35 e2 7f 00 00 	lea    0x7fe2(%rip),%rsi        # a219 <_IO_stdin_used+0x219>
    2237:	f2 48 0f 2a 84 24 a0 	cvtsi2sdq 0xa0(%rsp),%xmm0
    223e:	00 00 00 
    2241:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    2245:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    2249:	f2 0f 5e 44 24 30    	divsd  0x30(%rsp),%xmm0
    224f:	e8 cc f1 ff ff       	callq  1420 <__printf_chk@plt>
    2254:	66 0f ef c0          	pxor   %xmm0,%xmm0
    2258:	48 8b 05 81 84 00 00 	mov    0x8481(%rip),%rax        # a6e0 <__PRETTY_FUNCTION__.5230+0xf>
    225f:	f2 48 0f 2a 84 24 a0 	cvtsi2sdq 0xa0(%rsp),%xmm0
    2266:	00 00 00 
    2269:	48 8d 35 c4 7f 00 00 	lea    0x7fc4(%rip),%rsi        # a234 <_IO_stdin_used+0x234>
    2270:	bf 01 00 00 00       	mov    $0x1,%edi
    2275:	66 48 0f 6e c8       	movq   %rax,%xmm1
    227a:	b8 01 00 00 00       	mov    $0x1,%eax
    227f:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    2283:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
    2287:	f2 0f 5e 44 24 08    	divsd  0x8(%rsp),%xmm0
    228d:	e8 8e f1 ff ff       	callq  1420 <__printf_chk@plt>
    2292:	48 8b 84 24 88 01 00 	mov    0x188(%rsp),%rax
    2299:	00 
    229a:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    22a1:	00 00 
    22a3:	0f 85 3d 04 00 00    	jne    26e6 <main+0x1246>
    22a9:	48 81 c4 98 01 00 00 	add    $0x198,%rsp
    22b0:	31 c0                	xor    %eax,%eax
    22b2:	5b                   	pop    %rbx
    22b3:	5d                   	pop    %rbp
    22b4:	41 5c                	pop    %r12
    22b6:	41 5d                	pop    %r13
    22b8:	41 5e                	pop    %r14
    22ba:	41 5f                	pop    %r15
    22bc:	c3                   	retq   
    22bd:	8b b4 24 98 00 00 00 	mov    0x98(%rsp),%esi
    22c4:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    22cb:	e8 b0 0e 00 00       	callq  3180 <blaum_roth_coding_bitmatrix>
    22d0:	e9 f9 f9 ff ff       	jmpq   1cce <main+0x82e>
    22d5:	8b b4 24 98 00 00 00 	mov    0x98(%rsp),%esi
    22dc:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    22e3:	e8 e8 08 00 00       	callq  2bd0 <liberation_coding_bitmatrix>
    22e8:	e9 e1 f9 ff ff       	jmpq   1cce <main+0x82e>
    22ed:	48 8b 0d 4c ee 00 00 	mov    0xee4c(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    22f4:	ba 15 00 00 00       	mov    $0x15,%edx
    22f9:	be 01 00 00 00       	mov    $0x1,%esi
    22fe:	48 8d 3d 8d 7e 00 00 	lea    0x7e8d(%rip),%rdi        # a192 <_IO_stdin_used+0x192>
    2305:	e8 56 f1 ff ff       	callq  1460 <fwrite@plt>
    230a:	31 ff                	xor    %edi,%edi
    230c:	e8 3f f1 ff ff       	callq  1450 <exit@plt>
    2311:	8b 84 24 98 00 00 00 	mov    0x98(%rsp),%eax
    2318:	8d 50 f8             	lea    -0x8(%rax),%edx
    231b:	83 e2 f7             	and    $0xfffffff7,%edx
    231e:	74 09                	je     2329 <main+0xe89>
    2320:	83 f8 20             	cmp    $0x20,%eax
    2323:	0f 85 75 f9 ff ff    	jne    1c9e <main+0x7fe>
    2329:	c7 44 24 68 01 00 00 	movl   $0x1,0x68(%rsp)
    2330:	00 
    2331:	e9 7a f4 ff ff       	jmpq   17b0 <main+0x310>
    2336:	48 8d 35 ba 7d 00 00 	lea    0x7dba(%rip),%rsi        # a0f7 <_IO_stdin_used+0xf7>
    233d:	4c 89 e7             	mov    %r12,%rdi
    2340:	e8 3b f0 ff ff       	callq  1380 <strcmp@plt>
    2345:	85 c0                	test   %eax,%eax
    2347:	0f 85 e9 00 00 00    	jne    2436 <main+0xf96>
    234d:	83 bc 24 9c 00 00 00 	cmpl   $0x0,0x9c(%rsp)
    2354:	00 
    2355:	c7 44 24 68 02 00 00 	movl   $0x2,0x68(%rsp)
    235c:	00 
    235d:	0f 85 4d f4 ff ff    	jne    17b0 <main+0x310>
    2363:	48 8b 0d d6 ed 00 00 	mov    0xedd6(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    236a:	ba 19 00 00 00       	mov    $0x19,%edx
    236f:	be 01 00 00 00       	mov    $0x1,%esi
    2374:	48 8d 3d 88 7d 00 00 	lea    0x7d88(%rip),%rdi        # a103 <_IO_stdin_used+0x103>
    237b:	e8 e0 f0 ff ff       	callq  1460 <fwrite@plt>
    2380:	31 ff                	xor    %edi,%edi
    2382:	e8 c9 f0 ff ff       	callq  1450 <exit@plt>
    2387:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    238e:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    2395:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    239c:	e8 3f 73 00 00       	callq  96e0 <cauchy_good_general_coding_matrix>
    23a1:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    23a8:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    23af:	48 89 c1             	mov    %rax,%rcx
    23b2:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
    23b7:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    23be:	e8 fd 11 00 00       	callq  35c0 <jerasure_matrix_to_bitmatrix>
    23c3:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    23ca:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    23d1:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    23d8:	48 89 c1             	mov    %rax,%rcx
    23db:	e8 60 3e 00 00       	callq  6240 <jerasure_smart_bitmatrix_to_schedule>
    23e0:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    23e5:	e9 0f f9 ff ff       	jmpq   1cf9 <main+0x859>
    23ea:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    23f1:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    23f8:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    23ff:	e8 ac 6c 00 00       	callq  90b0 <reed_sol_vandermonde_coding_matrix>
    2404:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
    240b:	00 00 
    240d:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
    2412:	e9 e2 f8 ff ff       	jmpq   1cf9 <main+0x859>
    2417:	8b 94 24 98 00 00 00 	mov    0x98(%rsp),%edx
    241e:	8b b4 24 94 00 00 00 	mov    0x94(%rsp),%esi
    2425:	8b bc 24 90 00 00 00 	mov    0x90(%rsp),%edi
    242c:	e8 af 6e 00 00       	callq  92e0 <cauchy_original_coding_matrix>
    2431:	e9 6b ff ff ff       	jmpq   23a1 <main+0xf01>
    2436:	48 8d 35 e0 7c 00 00 	lea    0x7ce0(%rip),%rsi        # a11d <_IO_stdin_used+0x11d>
    243d:	4c 89 e7             	mov    %r12,%rdi
    2440:	e8 3b ef ff ff       	callq  1380 <strcmp@plt>
    2445:	85 c0                	test   %eax,%eax
    2447:	75 51                	jne    249a <main+0xffa>
    2449:	83 bc 24 9c 00 00 00 	cmpl   $0x0,0x9c(%rsp)
    2450:	00 
    2451:	0f 84 0c ff ff ff    	je     2363 <main+0xec3>
    2457:	c7 44 24 68 03 00 00 	movl   $0x3,0x68(%rsp)
    245e:	00 
    245f:	e9 4c f3 ff ff       	jmpq   17b0 <main+0x310>
    2464:	48 8d 0d 66 82 00 00 	lea    0x8266(%rip),%rcx        # a6d1 <__PRETTY_FUNCTION__.5230>
    246b:	ba 4c 01 00 00       	mov    $0x14c,%edx
    2470:	48 8d 35 b4 7b 00 00 	lea    0x7bb4(%rip),%rsi        # a02b <_IO_stdin_used+0x2b>
    2477:	48 8d 3d 6a 81 00 00 	lea    0x816a(%rip),%rdi        # a5e8 <_IO_stdin_used+0x5e8>
    247e:	e8 ad ee ff ff       	callq  1330 <__assert_fail@plt>
    2483:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
    248a:	00 00 
    248c:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
    2493:	00 00 
    2495:	e9 5f f8 ff ff       	jmpq   1cf9 <main+0x859>
    249a:	48 8d 35 88 7c 00 00 	lea    0x7c88(%rip),%rsi        # a129 <_IO_stdin_used+0x129>
    24a1:	4c 89 e7             	mov    %r12,%rdi
    24a4:	e8 d7 ee ff ff       	callq  1380 <strcmp@plt>
    24a9:	85 c0                	test   %eax,%eax
    24ab:	75 67                	jne    2514 <main+0x1074>
    24ad:	8b bc 24 98 00 00 00 	mov    0x98(%rsp),%edi
    24b4:	39 bc 24 90 00 00 00 	cmp    %edi,0x90(%rsp)
    24bb:	7f 33                	jg     24f0 <main+0x1050>
    24bd:	83 ff 02             	cmp    $0x2,%edi
    24c0:	7e 0a                	jle    24cc <main+0x102c>
    24c2:	40 f6 c7 01          	test   $0x1,%dil
    24c6:	0f 85 f6 00 00 00    	jne    25c2 <main+0x1122>
    24cc:	48 8b 0d 6d ec 00 00 	mov    0xec6d(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    24d3:	ba 2f 00 00 00       	mov    $0x2f,%edx
    24d8:	be 01 00 00 00       	mov    $0x1,%esi
    24dd:	48 8d 3d c4 7f 00 00 	lea    0x7fc4(%rip),%rdi        # a4a8 <_IO_stdin_used+0x4a8>
    24e4:	e8 77 ef ff ff       	callq  1460 <fwrite@plt>
    24e9:	31 ff                	xor    %edi,%edi
    24eb:	e8 60 ef ff ff       	callq  1450 <exit@plt>
    24f0:	48 8b 0d 49 ec 00 00 	mov    0xec49(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    24f7:	ba 22 00 00 00       	mov    $0x22,%edx
    24fc:	be 01 00 00 00       	mov    $0x1,%esi
    2501:	48 8d 3d 78 7f 00 00 	lea    0x7f78(%rip),%rdi        # a480 <_IO_stdin_used+0x480>
    2508:	e8 53 ef ff ff       	callq  1460 <fwrite@plt>
    250d:	31 ff                	xor    %edi,%edi
    250f:	e8 3c ef ff ff       	callq  1450 <exit@plt>
    2514:	48 8d 35 19 7c 00 00 	lea    0x7c19(%rip),%rsi        # a134 <_IO_stdin_used+0x134>
    251b:	4c 89 e7             	mov    %r12,%rdi
    251e:	e8 5d ee ff ff       	callq  1380 <strcmp@plt>
    2523:	85 c0                	test   %eax,%eax
    2525:	75 44                	jne    256b <main+0x10cb>
    2527:	8b 84 24 98 00 00 00 	mov    0x98(%rsp),%eax
    252e:	39 84 24 90 00 00 00 	cmp    %eax,0x90(%rsp)
    2535:	7f b9                	jg     24f0 <main+0x1050>
    2537:	83 f8 02             	cmp    $0x2,%eax
    253a:	7e 0b                	jle    2547 <main+0x10a7>
    253c:	8d 78 01             	lea    0x1(%rax),%edi
    253f:	a8 01                	test   $0x1,%al
    2541:	0f 84 cb 00 00 00    	je     2612 <main+0x1172>
    2547:	48 8b 0d f2 eb 00 00 	mov    0xebf2(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    254e:	ba 31 00 00 00       	mov    $0x31,%edx
    2553:	be 01 00 00 00       	mov    $0x1,%esi
    2558:	48 8d 3d a9 7f 00 00 	lea    0x7fa9(%rip),%rdi        # a508 <_IO_stdin_used+0x508>
    255f:	e8 fc ee ff ff       	callq  1460 <fwrite@plt>
    2564:	31 ff                	xor    %edi,%edi
    2566:	e8 e5 ee ff ff       	callq  1450 <exit@plt>
    256b:	48 8d 35 cd 7b 00 00 	lea    0x7bcd(%rip),%rsi        # a13f <_IO_stdin_used+0x13f>
    2572:	4c 89 e7             	mov    %r12,%rdi
    2575:	e8 06 ee ff ff       	callq  1380 <strcmp@plt>
    257a:	85 c0                	test   %eax,%eax
    257c:	0f 85 0f 01 00 00    	jne    2691 <main+0x11f1>
    2582:	83 bc 24 9c 00 00 00 	cmpl   $0x0,0x9c(%rsp)
    2589:	00 
    258a:	0f 84 dd 00 00 00    	je     266d <main+0x11cd>
    2590:	83 bc 24 98 00 00 00 	cmpl   $0x8,0x98(%rsp)
    2597:	08 
    2598:	0f 84 a1 00 00 00    	je     263f <main+0x119f>
    259e:	48 8b 0d 9b eb 00 00 	mov    0xeb9b(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    25a5:	ba 0f 00 00 00       	mov    $0xf,%edx
    25aa:	be 01 00 00 00       	mov    $0x1,%esi
    25af:	48 8d 3d ad 7b 00 00 	lea    0x7bad(%rip),%rdi        # a163 <_IO_stdin_used+0x163>
    25b6:	e8 a5 ee ff ff       	callq  1460 <fwrite@plt>
    25bb:	31 ff                	xor    %edi,%edi
    25bd:	e8 8e ee ff ff       	callq  1450 <exit@plt>
    25c2:	e8 a9 03 00 00       	callq  2970 <is_prime>
    25c7:	85 c0                	test   %eax,%eax
    25c9:	0f 84 fd fe ff ff    	je     24cc <main+0x102c>
    25cf:	8b 84 24 9c 00 00 00 	mov    0x9c(%rsp),%eax
    25d6:	85 c0                	test   %eax,%eax
    25d8:	0f 84 85 fd ff ff    	je     2363 <main+0xec3>
    25de:	c7 44 24 68 04 00 00 	movl   $0x4,0x68(%rsp)
    25e5:	00 
    25e6:	a8 07                	test   $0x7,%al
    25e8:	0f 84 c2 f1 ff ff    	je     17b0 <main+0x310>
    25ee:	48 8b 0d 4b eb 00 00 	mov    0xeb4b(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    25f5:	ba 2e 00 00 00       	mov    $0x2e,%edx
    25fa:	be 01 00 00 00       	mov    $0x1,%esi
    25ff:	48 8d 3d d2 7e 00 00 	lea    0x7ed2(%rip),%rdi        # a4d8 <_IO_stdin_used+0x4d8>
    2606:	e8 55 ee ff ff       	callq  1460 <fwrite@plt>
    260b:	31 ff                	xor    %edi,%edi
    260d:	e8 3e ee ff ff       	callq  1450 <exit@plt>
    2612:	e8 59 03 00 00       	callq  2970 <is_prime>
    2617:	85 c0                	test   %eax,%eax
    2619:	0f 84 28 ff ff ff    	je     2547 <main+0x10a7>
    261f:	8b 84 24 9c 00 00 00 	mov    0x9c(%rsp),%eax
    2626:	85 c0                	test   %eax,%eax
    2628:	0f 84 35 fd ff ff    	je     2363 <main+0xec3>
    262e:	a8 07                	test   $0x7,%al
    2630:	75 bc                	jne    25ee <main+0x114e>
    2632:	c7 44 24 68 05 00 00 	movl   $0x5,0x68(%rsp)
    2639:	00 
    263a:	e9 71 f1 ff ff       	jmpq   17b0 <main+0x310>
    263f:	83 bc 24 94 00 00 00 	cmpl   $0x2,0x94(%rsp)
    2646:	02 
    2647:	74 6c                	je     26b5 <main+0x1215>
    2649:	48 8b 0d f0 ea 00 00 	mov    0xeaf0(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    2650:	ba 0f 00 00 00       	mov    $0xf,%edx
    2655:	be 01 00 00 00       	mov    $0x1,%esi
    265a:	48 8d 3d 12 7b 00 00 	lea    0x7b12(%rip),%rdi        # a173 <_IO_stdin_used+0x173>
    2661:	e8 fa ed ff ff       	callq  1460 <fwrite@plt>
    2666:	31 ff                	xor    %edi,%edi
    2668:	e8 e3 ed ff ff       	callq  1450 <exit@plt>
    266d:	48 8b 0d cc ea 00 00 	mov    0xeacc(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    2674:	ba 18 00 00 00       	mov    $0x18,%edx
    2679:	be 01 00 00 00       	mov    $0x1,%esi
    267e:	48 8d 3d c5 7a 00 00 	lea    0x7ac5(%rip),%rdi        # a14a <_IO_stdin_used+0x14a>
    2685:	e8 d6 ed ff ff       	callq  1460 <fwrite@plt>
    268a:	31 ff                	xor    %edi,%edi
    268c:	e8 bf ed ff ff       	callq  1450 <exit@plt>
    2691:	48 8b 0d a8 ea 00 00 	mov    0xeaa8(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    2698:	ba a1 00 00 00       	mov    $0xa1,%edx
    269d:	be 01 00 00 00       	mov    $0x1,%esi
    26a2:	48 8d 3d 97 7e 00 00 	lea    0x7e97(%rip),%rdi        # a540 <_IO_stdin_used+0x540>
    26a9:	e8 b2 ed ff ff       	callq  1460 <fwrite@plt>
    26ae:	31 ff                	xor    %edi,%edi
    26b0:	e8 9b ed ff ff       	callq  1450 <exit@plt>
    26b5:	83 bc 24 90 00 00 00 	cmpl   $0x8,0x90(%rsp)
    26bc:	08 
    26bd:	0f 8f 2d fe ff ff    	jg     24f0 <main+0x1050>
    26c3:	c7 44 24 68 06 00 00 	movl   $0x6,0x68(%rsp)
    26ca:	00 
    26cb:	e9 e0 f0 ff ff       	jmpq   17b0 <main+0x310>
    26d0:	48 8d 3d e3 7a 00 00 	lea    0x7ae3(%rip),%rdi        # a1ba <_IO_stdin_used+0x1ba>
    26d7:	e8 64 ed ff ff       	callq  1440 <perror@plt>
    26dc:	bf 01 00 00 00       	mov    $0x1,%edi
    26e1:	e8 6a ed ff ff       	callq  1450 <exit@plt>
    26e6:	e8 15 ec ff ff       	callq  1300 <__stack_chk_fail@plt>
    26eb:	48 89 bc 24 a8 00 00 	mov    %rdi,0xa8(%rsp)
    26f2:	00 
    26f3:	e9 66 f0 ff ff       	jmpq   175e <main+0x2be>
    26f8:	48 8b 0d 41 ea 00 00 	mov    0xea41(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    26ff:	ba 43 00 00 00       	mov    $0x43,%edx
    2704:	be 01 00 00 00       	mov    $0x1,%esi
    2709:	48 8d 3d 28 7f 00 00 	lea    0x7f28(%rip),%rdi        # a638 <_IO_stdin_used+0x638>
    2710:	e8 4b ed ff ff       	callq  1460 <fwrite@plt>
    2715:	bf 01 00 00 00       	mov    $0x1,%edi
    271a:	e8 31 ed ff ff       	callq  1450 <exit@plt>
    271f:	90                   	nop

0000000000002720 <_start>:
    2720:	f3 0f 1e fa          	endbr64 
    2724:	31 ed                	xor    %ebp,%ebp
    2726:	49 89 d1             	mov    %rdx,%r9
    2729:	5e                   	pop    %rsi
    272a:	48 89 e2             	mov    %rsp,%rdx
    272d:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    2731:	50                   	push   %rax
    2732:	54                   	push   %rsp
    2733:	4c 8d 05 a6 73 00 00 	lea    0x73a6(%rip),%r8        # 9ae0 <__libc_csu_fini>
    273a:	48 8d 0d 2f 73 00 00 	lea    0x732f(%rip),%rcx        # 9a70 <__libc_csu_init>
    2741:	48 8d 3d 58 ed ff ff 	lea    -0x12a8(%rip),%rdi        # 14a0 <main>
    2748:	ff 15 92 b8 00 00    	callq  *0xb892(%rip)        # dfe0 <__libc_start_main@GLIBC_2.2.5>
    274e:	f4                   	hlt    
    274f:	90                   	nop

0000000000002750 <deregister_tm_clones>:
    2750:	48 8d 3d d1 e9 00 00 	lea    0xe9d1(%rip),%rdi        # 11128 <__TMC_END__>
    2757:	48 8d 05 ca e9 00 00 	lea    0xe9ca(%rip),%rax        # 11128 <__TMC_END__>
    275e:	48 39 f8             	cmp    %rdi,%rax
    2761:	74 15                	je     2778 <deregister_tm_clones+0x28>
    2763:	48 8b 05 6e b8 00 00 	mov    0xb86e(%rip),%rax        # dfd8 <_ITM_deregisterTMCloneTable>
    276a:	48 85 c0             	test   %rax,%rax
    276d:	74 09                	je     2778 <deregister_tm_clones+0x28>
    276f:	ff e0                	jmpq   *%rax
    2771:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2778:	c3                   	retq   
    2779:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002780 <register_tm_clones>:
    2780:	48 8d 3d a1 e9 00 00 	lea    0xe9a1(%rip),%rdi        # 11128 <__TMC_END__>
    2787:	48 8d 35 9a e9 00 00 	lea    0xe99a(%rip),%rsi        # 11128 <__TMC_END__>
    278e:	48 29 fe             	sub    %rdi,%rsi
    2791:	48 89 f0             	mov    %rsi,%rax
    2794:	48 c1 ee 3f          	shr    $0x3f,%rsi
    2798:	48 c1 f8 03          	sar    $0x3,%rax
    279c:	48 01 c6             	add    %rax,%rsi
    279f:	48 d1 fe             	sar    %rsi
    27a2:	74 14                	je     27b8 <register_tm_clones+0x38>
    27a4:	48 8b 05 45 b8 00 00 	mov    0xb845(%rip),%rax        # dff0 <_ITM_registerTMCloneTable>
    27ab:	48 85 c0             	test   %rax,%rax
    27ae:	74 08                	je     27b8 <register_tm_clones+0x38>
    27b0:	ff e0                	jmpq   *%rax
    27b2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    27b8:	c3                   	retq   
    27b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000027c0 <__do_global_dtors_aux>:
    27c0:	f3 0f 1e fa          	endbr64 
    27c4:	80 3d 7d e9 00 00 00 	cmpb   $0x0,0xe97d(%rip)        # 11148 <completed.8061>
    27cb:	75 2b                	jne    27f8 <__do_global_dtors_aux+0x38>
    27cd:	55                   	push   %rbp
    27ce:	48 83 3d 22 b8 00 00 	cmpq   $0x0,0xb822(%rip)        # dff8 <__cxa_finalize@GLIBC_2.2.5>
    27d5:	00 
    27d6:	48 89 e5             	mov    %rsp,%rbp
    27d9:	74 0c                	je     27e7 <__do_global_dtors_aux+0x27>
    27db:	48 8b 3d 26 b8 00 00 	mov    0xb826(%rip),%rdi        # e008 <__dso_handle>
    27e2:	e8 79 ea ff ff       	callq  1260 <__cxa_finalize@plt>
    27e7:	e8 64 ff ff ff       	callq  2750 <deregister_tm_clones>
    27ec:	c6 05 55 e9 00 00 01 	movb   $0x1,0xe955(%rip)        # 11148 <completed.8061>
    27f3:	5d                   	pop    %rbp
    27f4:	c3                   	retq   
    27f5:	0f 1f 00             	nopl   (%rax)
    27f8:	c3                   	retq   
    27f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002800 <frame_dummy>:
    2800:	f3 0f 1e fa          	endbr64 
    2804:	e9 77 ff ff ff       	jmpq   2780 <register_tm_clones>
    2809:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002810 <ctrl_bs_handler>:
    2810:	f3 0f 1e fa          	endbr64 
    2814:	48 83 ec 18          	sub    $0x18,%rsp
    2818:	31 ff                	xor    %edi,%edi
    281a:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2821:	00 00 
    2823:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    2828:	31 c0                	xor    %eax,%eax
    282a:	e8 91 eb ff ff       	callq  13c0 <time@plt>
    282f:	48 89 e7             	mov    %rsp,%rdi
    2832:	48 89 04 24          	mov    %rax,(%rsp)
    2836:	e8 b5 ea ff ff       	callq  12f0 <ctime@plt>
    283b:	48 8b 3d fe e8 00 00 	mov    0xe8fe(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    2842:	be 01 00 00 00       	mov    $0x1,%esi
    2847:	48 8d 15 b6 77 00 00 	lea    0x77b6(%rip),%rdx        # a004 <_IO_stdin_used+0x4>
    284e:	48 89 c1             	mov    %rax,%rcx
    2851:	31 c0                	xor    %eax,%eax
    2853:	e8 18 ec ff ff       	callq  1470 <__fprintf_chk@plt>
    2858:	ba 24 00 00 00       	mov    $0x24,%edx
    285d:	48 8b 0d dc e8 00 00 	mov    0xe8dc(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    2864:	be 01 00 00 00       	mov    $0x1,%esi
    2869:	48 8d 3d e0 79 00 00 	lea    0x79e0(%rip),%rdi        # a250 <_IO_stdin_used+0x250>
    2870:	e8 eb eb ff ff       	callq  1460 <fwrite@plt>
    2875:	8b 0d 95 00 01 00    	mov    0x10095(%rip),%ecx        # 12910 <readins>
    287b:	be 01 00 00 00       	mov    $0x1,%esi
    2880:	31 c0                	xor    %eax,%eax
    2882:	48 8b 3d b7 e8 00 00 	mov    0xe8b7(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    2889:	48 8d 15 e8 79 00 00 	lea    0x79e8(%rip),%rdx        # a278 <_IO_stdin_used+0x278>
    2890:	e8 db eb ff ff       	callq  1470 <__fprintf_chk@plt>
    2895:	8b 0d 79 00 01 00    	mov    0x10079(%rip),%ecx        # 12914 <n>
    289b:	be 01 00 00 00       	mov    $0x1,%esi
    28a0:	31 c0                	xor    %eax,%eax
    28a2:	48 8b 3d 97 e8 00 00 	mov    0xe897(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    28a9:	48 8d 15 59 77 00 00 	lea    0x7759(%rip),%rdx        # a009 <_IO_stdin_used+0x9>
    28b0:	e8 bb eb ff ff       	callq  1470 <__fprintf_chk@plt>
    28b5:	8b 15 5d 00 01 00    	mov    0x1005d(%rip),%edx        # 12918 <method>
    28bb:	be 01 00 00 00       	mov    $0x1,%esi
    28c0:	48 8d 05 59 b7 00 00 	lea    0xb759(%rip),%rax        # e020 <Methods>
    28c7:	48 8b 3d 72 e8 00 00 	mov    0xe872(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    28ce:	48 8b 0c d0          	mov    (%rax,%rdx,8),%rcx
    28d2:	48 8d 15 45 77 00 00 	lea    0x7745(%rip),%rdx        # a01e <_IO_stdin_used+0x1e>
    28d9:	31 c0                	xor    %eax,%eax
    28db:	e8 90 eb ff ff       	callq  1470 <__fprintf_chk@plt>
    28e0:	48 8d 35 29 ff ff ff 	lea    -0xd7(%rip),%rsi        # 2810 <ctrl_bs_handler>
    28e7:	bf 03 00 00 00       	mov    $0x3,%edi
    28ec:	e8 9f ea ff ff       	callq  1390 <signal@plt>
    28f1:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    28f6:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    28fd:	00 00 
    28ff:	75 05                	jne    2906 <ctrl_bs_handler+0xf6>
    2901:	48 83 c4 18          	add    $0x18,%rsp
    2905:	c3                   	retq   
    2906:	e8 f5 e9 ff ff       	callq  1300 <__stack_chk_fail@plt>
    290b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002910 <jfread>:
    2910:	f3 0f 1e fa          	endbr64 
    2914:	41 54                	push   %r12
    2916:	48 63 f6             	movslq %esi,%rsi
    2919:	55                   	push   %rbp
    291a:	53                   	push   %rbx
    291b:	48 85 c9             	test   %rcx,%rcx
    291e:	74 10                	je     2930 <jfread+0x20>
    2920:	e8 9b e9 ff ff       	callq  12c0 <fread@plt>
    2925:	5b                   	pop    %rbx
    2926:	5d                   	pop    %rbp
    2927:	41 5c                	pop    %r12
    2929:	c3                   	retq   
    292a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2930:	48 89 f5             	mov    %rsi,%rbp
    2933:	48 c1 ee 02          	shr    $0x2,%rsi
    2937:	85 f6                	test   %esi,%esi
    2939:	7e 26                	jle    2961 <jfread+0x51>
    293b:	8d 46 ff             	lea    -0x1(%rsi),%eax
    293e:	48 89 fb             	mov    %rdi,%rbx
    2941:	4c 8d 64 87 04       	lea    0x4(%rdi,%rax,4),%r12
    2946:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    294d:	00 00 00 
    2950:	e8 0b ea ff ff       	callq  1360 <mrand48@plt>
    2955:	48 83 c3 04          	add    $0x4,%rbx
    2959:	89 43 fc             	mov    %eax,-0x4(%rbx)
    295c:	49 39 dc             	cmp    %rbx,%r12
    295f:	75 ef                	jne    2950 <jfread+0x40>
    2961:	89 e8                	mov    %ebp,%eax
    2963:	5b                   	pop    %rbx
    2964:	5d                   	pop    %rbp
    2965:	41 5c                	pop    %r12
    2967:	c3                   	retq   
    2968:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    296f:	00 

0000000000002970 <is_prime>:
    2970:	f3 0f 1e fa          	endbr64 
    2974:	48 81 ec f8 00 00 00 	sub    $0xf8,%rsp
    297b:	be 02 00 00 00       	mov    $0x2,%esi
    2980:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2987:	00 00 
    2989:	48 89 84 24 e8 00 00 	mov    %rax,0xe8(%rsp)
    2990:	00 
    2991:	31 c0                	xor    %eax,%eax
    2993:	48 8d 4c 24 04       	lea    0x4(%rsp),%rcx
    2998:	4c 8d 84 24 dc 00 00 	lea    0xdc(%rsp),%r8
    299f:	00 
    29a0:	48 b8 03 00 00 00 05 	movabs $0x500000003,%rax
    29a7:	00 00 00 
    29aa:	48 89 44 24 04       	mov    %rax,0x4(%rsp)
    29af:	48 b8 07 00 00 00 0b 	movabs $0xb00000007,%rax
    29b6:	00 00 00 
    29b9:	48 89 44 24 0c       	mov    %rax,0xc(%rsp)
    29be:	48 b8 0d 00 00 00 11 	movabs $0x110000000d,%rax
    29c5:	00 00 00 
    29c8:	48 89 44 24 14       	mov    %rax,0x14(%rsp)
    29cd:	48 b8 13 00 00 00 17 	movabs $0x1700000013,%rax
    29d4:	00 00 00 
    29d7:	48 89 44 24 1c       	mov    %rax,0x1c(%rsp)
    29dc:	48 b8 1d 00 00 00 1f 	movabs $0x1f0000001d,%rax
    29e3:	00 00 00 
    29e6:	48 89 44 24 24       	mov    %rax,0x24(%rsp)
    29eb:	48 b8 25 00 00 00 29 	movabs $0x2900000025,%rax
    29f2:	00 00 00 
    29f5:	48 89 44 24 2c       	mov    %rax,0x2c(%rsp)
    29fa:	48 b8 2b 00 00 00 2f 	movabs $0x2f0000002b,%rax
    2a01:	00 00 00 
    2a04:	48 89 44 24 34       	mov    %rax,0x34(%rsp)
    2a09:	48 b8 35 00 00 00 3b 	movabs $0x3b00000035,%rax
    2a10:	00 00 00 
    2a13:	48 89 44 24 3c       	mov    %rax,0x3c(%rsp)
    2a18:	48 b8 3d 00 00 00 43 	movabs $0x430000003d,%rax
    2a1f:	00 00 00 
    2a22:	48 89 44 24 44       	mov    %rax,0x44(%rsp)
    2a27:	48 b8 47 00 00 00 49 	movabs $0x4900000047,%rax
    2a2e:	00 00 00 
    2a31:	48 89 44 24 4c       	mov    %rax,0x4c(%rsp)
    2a36:	48 b8 4f 00 00 00 53 	movabs $0x530000004f,%rax
    2a3d:	00 00 00 
    2a40:	48 89 44 24 54       	mov    %rax,0x54(%rsp)
    2a45:	48 b8 59 00 00 00 61 	movabs $0x6100000059,%rax
    2a4c:	00 00 00 
    2a4f:	48 89 44 24 5c       	mov    %rax,0x5c(%rsp)
    2a54:	48 b8 65 00 00 00 67 	movabs $0x6700000065,%rax
    2a5b:	00 00 00 
    2a5e:	48 89 44 24 64       	mov    %rax,0x64(%rsp)
    2a63:	48 b8 6b 00 00 00 6d 	movabs $0x6d0000006b,%rax
    2a6a:	00 00 00 
    2a6d:	48 89 44 24 6c       	mov    %rax,0x6c(%rsp)
    2a72:	48 b8 71 00 00 00 7f 	movabs $0x7f00000071,%rax
    2a79:	00 00 00 
    2a7c:	48 89 44 24 74       	mov    %rax,0x74(%rsp)
    2a81:	48 b8 83 00 00 00 89 	movabs $0x8900000083,%rax
    2a88:	00 00 00 
    2a8b:	48 89 44 24 7c       	mov    %rax,0x7c(%rsp)
    2a90:	48 b8 8b 00 00 00 95 	movabs $0x950000008b,%rax
    2a97:	00 00 00 
    2a9a:	48 89 84 24 84 00 00 	mov    %rax,0x84(%rsp)
    2aa1:	00 
    2aa2:	48 b8 97 00 00 00 9d 	movabs $0x9d00000097,%rax
    2aa9:	00 00 00 
    2aac:	48 89 84 24 8c 00 00 	mov    %rax,0x8c(%rsp)
    2ab3:	00 
    2ab4:	48 b8 a3 00 00 00 a7 	movabs $0xa7000000a3,%rax
    2abb:	00 00 00 
    2abe:	48 89 84 24 94 00 00 	mov    %rax,0x94(%rsp)
    2ac5:	00 
    2ac6:	48 b8 ad 00 00 00 b3 	movabs $0xb3000000ad,%rax
    2acd:	00 00 00 
    2ad0:	48 89 84 24 9c 00 00 	mov    %rax,0x9c(%rsp)
    2ad7:	00 
    2ad8:	48 b8 b5 00 00 00 bf 	movabs $0xbf000000b5,%rax
    2adf:	00 00 00 
    2ae2:	48 89 84 24 a4 00 00 	mov    %rax,0xa4(%rsp)
    2ae9:	00 
    2aea:	48 b8 c1 00 00 00 c5 	movabs $0xc5000000c1,%rax
    2af1:	00 00 00 
    2af4:	48 89 84 24 ac 00 00 	mov    %rax,0xac(%rsp)
    2afb:	00 
    2afc:	48 b8 c7 00 00 00 d3 	movabs $0xd3000000c7,%rax
    2b03:	00 00 00 
    2b06:	48 89 84 24 b4 00 00 	mov    %rax,0xb4(%rsp)
    2b0d:	00 
    2b0e:	48 b8 df 00 00 00 e3 	movabs $0xe3000000df,%rax
    2b15:	00 00 00 
    2b18:	48 89 84 24 bc 00 00 	mov    %rax,0xbc(%rsp)
    2b1f:	00 
    2b20:	48 b8 e5 00 00 00 e9 	movabs $0xe9000000e5,%rax
    2b27:	00 00 00 
    2b2a:	48 89 84 24 c4 00 00 	mov    %rax,0xc4(%rsp)
    2b31:	00 
    2b32:	48 b8 ef 00 00 00 f1 	movabs $0xf1000000ef,%rax
    2b39:	00 00 00 
    2b3c:	48 89 84 24 cc 00 00 	mov    %rax,0xcc(%rsp)
    2b43:	00 
    2b44:	48 b8 fb 00 00 00 01 	movabs $0x101000000fb,%rax
    2b4b:	01 00 00 
    2b4e:	48 89 84 24 d4 00 00 	mov    %rax,0xd4(%rsp)
    2b55:	00 
    2b56:	eb 13                	jmp    2b6b <is_prime+0x1fb>
    2b58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2b5f:	00 
    2b60:	4c 39 c1             	cmp    %r8,%rcx
    2b63:	74 3b                	je     2ba0 <is_prime+0x230>
    2b65:	8b 31                	mov    (%rcx),%esi
    2b67:	48 83 c1 04          	add    $0x4,%rcx
    2b6b:	89 f8                	mov    %edi,%eax
    2b6d:	99                   	cltd   
    2b6e:	f7 fe                	idiv   %esi
    2b70:	85 d2                	test   %edx,%edx
    2b72:	75 ec                	jne    2b60 <is_prime+0x1f0>
    2b74:	31 c0                	xor    %eax,%eax
    2b76:	39 f7                	cmp    %esi,%edi
    2b78:	0f 94 c0             	sete   %al
    2b7b:	48 8b bc 24 e8 00 00 	mov    0xe8(%rsp),%rdi
    2b82:	00 
    2b83:	64 48 33 3c 25 28 00 	xor    %fs:0x28,%rdi
    2b8a:	00 00 
    2b8c:	75 31                	jne    2bbf <is_prime+0x24f>
    2b8e:	48 81 c4 f8 00 00 00 	add    $0xf8,%rsp
    2b95:	c3                   	retq   
    2b96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    2b9d:	00 00 00 
    2ba0:	48 8d 0d 21 7b 00 00 	lea    0x7b21(%rip),%rcx        # a6c8 <__PRETTY_FUNCTION__.5291>
    2ba7:	ba 65 02 00 00       	mov    $0x265,%edx
    2bac:	48 8d 35 78 74 00 00 	lea    0x7478(%rip),%rsi        # a02b <_IO_stdin_used+0x2b>
    2bb3:	48 8d 3d 7b 74 00 00 	lea    0x747b(%rip),%rdi        # a035 <_IO_stdin_used+0x35>
    2bba:	e8 71 e7 ff ff       	callq  1330 <__assert_fail@plt>
    2bbf:	e8 3c e7 ff ff       	callq  1300 <__stack_chk_fail@plt>
    2bc4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    2bcb:	00 00 00 
    2bce:	66 90                	xchg   %ax,%ax

0000000000002bd0 <liberation_coding_bitmatrix>:
    2bd0:	f3 0f 1e fa          	endbr64 
    2bd4:	39 f7                	cmp    %esi,%edi
    2bd6:	0f 8f 50 01 00 00    	jg     2d2c <liberation_coding_bitmatrix+0x15c>
    2bdc:	41 56                	push   %r14
    2bde:	41 55                	push   %r13
    2be0:	41 54                	push   %r12
    2be2:	41 89 fc             	mov    %edi,%r12d
    2be5:	44 0f af e6          	imul   %esi,%r12d
    2be9:	55                   	push   %rbp
    2bea:	89 fd                	mov    %edi,%ebp
    2bec:	53                   	push   %rbx
    2bed:	89 f3                	mov    %esi,%ebx
    2bef:	45 89 e5             	mov    %r12d,%r13d
    2bf2:	44 0f af ee          	imul   %esi,%r13d
    2bf6:	43 8d 7c 2d 00       	lea    0x0(%r13,%r13,1),%edi
    2bfb:	48 63 ff             	movslq %edi,%rdi
    2bfe:	48 c1 e7 02          	shl    $0x2,%rdi
    2c02:	e8 e9 e7 ff ff       	callq  13f0 <malloc@plt>
    2c07:	49 89 c0             	mov    %rax,%r8
    2c0a:	48 85 c0             	test   %rax,%rax
    2c0d:	0f 84 0a 01 00 00    	je     2d1d <liberation_coding_bitmatrix+0x14d>
    2c13:	4c 63 f3             	movslq %ebx,%r14
    2c16:	48 63 c5             	movslq %ebp,%rax
    2c19:	4c 89 c7             	mov    %r8,%rdi
    2c1c:	31 f6                	xor    %esi,%esi
    2c1e:	4c 89 f2             	mov    %r14,%rdx
    2c21:	49 0f af d6          	imul   %r14,%rdx
    2c25:	48 0f af d0          	imul   %rax,%rdx
    2c29:	48 c1 e2 03          	shl    $0x3,%rdx
    2c2d:	e8 0e e7 ff ff       	callq  1340 <memset@plt>
    2c32:	49 89 c0             	mov    %rax,%r8
    2c35:	85 db                	test   %ebx,%ebx
    2c37:	7e 49                	jle    2c82 <liberation_coding_bitmatrix+0xb2>
    2c39:	49 63 c4             	movslq %r12d,%rax
    2c3c:	4c 89 c7             	mov    %r8,%rdi
    2c3f:	4a 8d 0c b5 00 00 00 	lea    0x0(,%r14,4),%rcx
    2c46:	00 
    2c47:	31 f6                	xor    %esi,%esi
    2c49:	4c 8d 0c 85 04 00 00 	lea    0x4(,%rax,4),%r9
    2c50:	00 
    2c51:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2c58:	85 ed                	test   %ebp,%ebp
    2c5a:	7e 1c                	jle    2c78 <liberation_coding_bitmatrix+0xa8>
    2c5c:	48 89 fa             	mov    %rdi,%rdx
    2c5f:	31 c0                	xor    %eax,%eax
    2c61:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2c68:	83 c0 01             	add    $0x1,%eax
    2c6b:	c7 02 01 00 00 00    	movl   $0x1,(%rdx)
    2c71:	48 01 ca             	add    %rcx,%rdx
    2c74:	39 c5                	cmp    %eax,%ebp
    2c76:	75 f0                	jne    2c68 <liberation_coding_bitmatrix+0x98>
    2c78:	83 c6 01             	add    $0x1,%esi
    2c7b:	4c 01 cf             	add    %r9,%rdi
    2c7e:	39 f3                	cmp    %esi,%ebx
    2c80:	75 d6                	jne    2c58 <liberation_coding_bitmatrix+0x88>
    2c82:	85 ed                	test   %ebp,%ebp
    2c84:	0f 8e 87 00 00 00    	jle    2d11 <liberation_coding_bitmatrix+0x141>
    2c8a:	8d 43 ff             	lea    -0x1(%rbx),%eax
    2c8d:	45 89 eb             	mov    %r13d,%r11d
    2c90:	89 df                	mov    %ebx,%edi
    2c92:	45 31 ed             	xor    %r13d,%r13d
    2c95:	41 89 c6             	mov    %eax,%r14d
    2c98:	45 31 c9             	xor    %r9d,%r9d
    2c9b:	41 c1 ee 1f          	shr    $0x1f,%r14d
    2c9f:	41 01 c6             	add    %eax,%r14d
    2ca2:	41 d1 fe             	sar    %r14d
    2ca5:	0f 1f 00             	nopl   (%rax)
    2ca8:	85 db                	test   %ebx,%ebx
    2caa:	7e 28                	jle    2cd4 <liberation_coding_bitmatrix+0x104>
    2cac:	44 89 c9             	mov    %r9d,%ecx
    2caf:	44 89 de             	mov    %r11d,%esi
    2cb2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2cb8:	89 c8                	mov    %ecx,%eax
    2cba:	83 c1 01             	add    $0x1,%ecx
    2cbd:	99                   	cltd   
    2cbe:	f7 fb                	idiv   %ebx
    2cc0:	01 f2                	add    %esi,%edx
    2cc2:	44 01 e6             	add    %r12d,%esi
    2cc5:	48 63 d2             	movslq %edx,%rdx
    2cc8:	41 c7 04 90 01 00 00 	movl   $0x1,(%r8,%rdx,4)
    2ccf:	00 
    2cd0:	39 cf                	cmp    %ecx,%edi
    2cd2:	75 e4                	jne    2cb8 <liberation_coding_bitmatrix+0xe8>
    2cd4:	45 85 c9             	test   %r9d,%r9d
    2cd7:	74 26                	je     2cff <liberation_coding_bitmatrix+0x12f>
    2cd9:	44 89 e8             	mov    %r13d,%eax
    2cdc:	89 e9                	mov    %ebp,%ecx
    2cde:	99                   	cltd   
    2cdf:	f7 fb                	idiv   %ebx
    2ce1:	0f af ca             	imul   %edx,%ecx
    2ce4:	41 8d 44 11 ff       	lea    -0x1(%r9,%rdx,1),%eax
    2ce9:	99                   	cltd   
    2cea:	f7 fb                	idiv   %ebx
    2cec:	0f af cb             	imul   %ebx,%ecx
    2cef:	44 01 d9             	add    %r11d,%ecx
    2cf2:	01 d1                	add    %edx,%ecx
    2cf4:	48 63 c9             	movslq %ecx,%rcx
    2cf7:	41 c7 04 88 01 00 00 	movl   $0x1,(%r8,%rcx,4)
    2cfe:	00 
    2cff:	41 83 c1 01          	add    $0x1,%r9d
    2d03:	45 01 f5             	add    %r14d,%r13d
    2d06:	41 01 db             	add    %ebx,%r11d
    2d09:	83 c7 01             	add    $0x1,%edi
    2d0c:	44 39 cd             	cmp    %r9d,%ebp
    2d0f:	75 97                	jne    2ca8 <liberation_coding_bitmatrix+0xd8>
    2d11:	5b                   	pop    %rbx
    2d12:	4c 89 c0             	mov    %r8,%rax
    2d15:	5d                   	pop    %rbp
    2d16:	41 5c                	pop    %r12
    2d18:	41 5d                	pop    %r13
    2d1a:	41 5e                	pop    %r14
    2d1c:	c3                   	retq   
    2d1d:	45 31 c0             	xor    %r8d,%r8d
    2d20:	5b                   	pop    %rbx
    2d21:	5d                   	pop    %rbp
    2d22:	4c 89 c0             	mov    %r8,%rax
    2d25:	41 5c                	pop    %r12
    2d27:	41 5d                	pop    %r13
    2d29:	41 5e                	pop    %r14
    2d2b:	c3                   	retq   
    2d2c:	45 31 c0             	xor    %r8d,%r8d
    2d2f:	4c 89 c0             	mov    %r8,%rax
    2d32:	c3                   	retq   
    2d33:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    2d3a:	00 00 00 00 
    2d3e:	66 90                	xchg   %ax,%ax

0000000000002d40 <liber8tion_coding_bitmatrix>:
    2d40:	f3 0f 1e fa          	endbr64 
    2d44:	83 ff 08             	cmp    $0x8,%edi
    2d47:	0f 8f 27 04 00 00    	jg     3174 <liber8tion_coding_bitmatrix+0x434>
    2d4d:	41 54                	push   %r12
    2d4f:	be 01 00 00 00       	mov    $0x1,%esi
    2d54:	55                   	push   %rbp
    2d55:	53                   	push   %rbx
    2d56:	89 fb                	mov    %edi,%ebx
    2d58:	c1 e7 07             	shl    $0x7,%edi
    2d5b:	48 63 ff             	movslq %edi,%rdi
    2d5e:	48 c1 e7 02          	shl    $0x2,%rdi
    2d62:	e8 09 e6 ff ff       	callq  1370 <calloc@plt>
    2d67:	48 85 c0             	test   %rax,%rax
    2d6a:	0f 84 fd 03 00 00    	je     316d <liber8tion_coding_bitmatrix+0x42d>
    2d70:	8d 7b ff             	lea    -0x1(%rbx),%edi
    2d73:	44 8d 0c dd 00 00 00 	lea    0x0(,%rbx,8),%r9d
    2d7a:	00 
    2d7b:	be 08 00 00 00       	mov    $0x8,%esi
    2d80:	48 83 c7 01          	add    $0x1,%rdi
    2d84:	49 63 d1             	movslq %r9d,%rdx
    2d87:	48 89 f9             	mov    %rdi,%rcx
    2d8a:	48 f7 df             	neg    %rdi
    2d8d:	4c 8d 04 95 04 00 00 	lea    0x4(,%rdx,4),%r8
    2d94:	00 
    2d95:	48 c1 e1 05          	shl    $0x5,%rcx
    2d99:	48 c1 e7 05          	shl    $0x5,%rdi
    2d9d:	48 01 c1             	add    %rax,%rcx
    2da0:	85 db                	test   %ebx,%ebx
    2da2:	7e 1b                	jle    2dbf <liber8tion_coding_bitmatrix+0x7f>
    2da4:	48 8d 14 0f          	lea    (%rdi,%rcx,1),%rdx
    2da8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2daf:	00 
    2db0:	c7 02 01 00 00 00    	movl   $0x1,(%rdx)
    2db6:	48 83 c2 20          	add    $0x20,%rdx
    2dba:	48 39 ca             	cmp    %rcx,%rdx
    2dbd:	75 f1                	jne    2db0 <liber8tion_coding_bitmatrix+0x70>
    2dbf:	4c 01 c1             	add    %r8,%rcx
    2dc2:	83 ee 01             	sub    $0x1,%esi
    2dc5:	75 d9                	jne    2da0 <liber8tion_coding_bitmatrix+0x60>
    2dc7:	85 db                	test   %ebx,%ebx
    2dc9:	0f 84 99 03 00 00    	je     3168 <liber8tion_coding_bitmatrix+0x428>
    2dcf:	41 89 d8             	mov    %ebx,%r8d
    2dd2:	41 c1 e0 06          	shl    $0x6,%r8d
    2dd6:	43 8d 3c 08          	lea    (%r8,%r9,1),%edi
    2dda:	49 63 d0             	movslq %r8d,%rdx
    2ddd:	c7 04 90 01 00 00 00 	movl   $0x1,(%rax,%rdx,4)
    2de4:	46 8d 24 0f          	lea    (%rdi,%r9,1),%r12d
    2de8:	48 63 d7             	movslq %edi,%rdx
    2deb:	c7 44 90 04 01 00 00 	movl   $0x1,0x4(%rax,%rdx,4)
    2df2:	00 
    2df3:	47 8d 1c 0c          	lea    (%r12,%r9,1),%r11d
    2df7:	49 63 d4             	movslq %r12d,%rdx
    2dfa:	c7 44 90 08 01 00 00 	movl   $0x1,0x8(%rax,%rdx,4)
    2e01:	00 
    2e02:	43 8d 34 0b          	lea    (%r11,%r9,1),%esi
    2e06:	49 63 d3             	movslq %r11d,%rdx
    2e09:	c7 44 90 0c 01 00 00 	movl   $0x1,0xc(%rax,%rdx,4)
    2e10:	00 
    2e11:	42 8d 0c 0e          	lea    (%rsi,%r9,1),%ecx
    2e15:	48 63 d6             	movslq %esi,%rdx
    2e18:	c7 44 90 10 01 00 00 	movl   $0x1,0x10(%rax,%rdx,4)
    2e1f:	00 
    2e20:	48 63 d1             	movslq %ecx,%rdx
    2e23:	c7 44 90 14 01 00 00 	movl   $0x1,0x14(%rax,%rdx,4)
    2e2a:	00 
    2e2b:	42 8d 14 09          	lea    (%rcx,%r9,1),%edx
    2e2f:	48 63 ea             	movslq %edx,%rbp
    2e32:	41 01 d1             	add    %edx,%r9d
    2e35:	c7 44 a8 18 01 00 00 	movl   $0x1,0x18(%rax,%rbp,4)
    2e3c:	00 
    2e3d:	49 63 e9             	movslq %r9d,%rbp
    2e40:	c7 44 a8 1c 01 00 00 	movl   $0x1,0x1c(%rax,%rbp,4)
    2e47:	00 
    2e48:	83 fb 01             	cmp    $0x1,%ebx
    2e4b:	0f 84 17 03 00 00    	je     3168 <liber8tion_coding_bitmatrix+0x428>
    2e51:	41 83 c0 08          	add    $0x8,%r8d
    2e55:	41 83 c4 08          	add    $0x8,%r12d
    2e59:	83 c1 08             	add    $0x8,%ecx
    2e5c:	83 c2 08             	add    $0x8,%edx
    2e5f:	4d 63 c0             	movslq %r8d,%r8
    2e62:	4d 63 e4             	movslq %r12d,%r12
    2e65:	48 63 c9             	movslq %ecx,%rcx
    2e68:	48 63 d2             	movslq %edx,%rdx
    2e6b:	49 83 c0 07          	add    $0x7,%r8
    2e6f:	48 83 c1 01          	add    $0x1,%rcx
    2e73:	48 83 c2 05          	add    $0x5,%rdx
    2e77:	42 c7 04 80 01 00 00 	movl   $0x1,(%rax,%r8,4)
    2e7e:	00 
    2e7f:	4a 8d 2c 85 00 00 00 	lea    0x0(,%r8,4),%rbp
    2e86:	00 
    2e87:	44 8d 47 08          	lea    0x8(%rdi),%r8d
    2e8b:	4d 63 c0             	movslq %r8d,%r8
    2e8e:	49 83 c0 03          	add    $0x3,%r8
    2e92:	42 c7 04 80 01 00 00 	movl   $0x1,(%rax,%r8,4)
    2e99:	00 
    2e9a:	4a 8d 3c 85 00 00 00 	lea    0x0(,%r8,4),%rdi
    2ea1:	00 
    2ea2:	4e 8d 04 a5 00 00 00 	lea    0x0(,%r12,4),%r8
    2ea9:	00 
    2eaa:	42 c7 04 a0 01 00 00 	movl   $0x1,(%rax,%r12,4)
    2eb1:	00 
    2eb2:	45 8d 63 08          	lea    0x8(%r11),%r12d
    2eb6:	4d 63 e4             	movslq %r12d,%r12
    2eb9:	49 83 c4 02          	add    $0x2,%r12
    2ebd:	42 c7 04 a0 01 00 00 	movl   $0x1,(%rax,%r12,4)
    2ec4:	00 
    2ec5:	4e 8d 1c a5 00 00 00 	lea    0x0(,%r12,4),%r11
    2ecc:	00 
    2ecd:	44 8d 66 08          	lea    0x8(%rsi),%r12d
    2ed1:	4d 63 e4             	movslq %r12d,%r12
    2ed4:	49 83 c4 06          	add    $0x6,%r12
    2ed8:	42 c7 04 a0 01 00 00 	movl   $0x1,(%rax,%r12,4)
    2edf:	00 
    2ee0:	4a 8d 34 a5 00 00 00 	lea    0x0(,%r12,4),%rsi
    2ee7:	00 
    2ee8:	4c 8d 24 8d 00 00 00 	lea    0x0(,%rcx,4),%r12
    2eef:	00 
    2ef0:	c7 04 88 01 00 00 00 	movl   $0x1,(%rax,%rcx,4)
    2ef7:	48 8d 0c 95 00 00 00 	lea    0x0(,%rdx,4),%rcx
    2efe:	00 
    2eff:	c7 04 90 01 00 00 00 	movl   $0x1,(%rax,%rdx,4)
    2f06:	41 8d 51 08          	lea    0x8(%r9),%edx
    2f0a:	48 63 d2             	movslq %edx,%rdx
    2f0d:	48 83 c2 04          	add    $0x4,%rdx
    2f11:	c7 04 90 01 00 00 00 	movl   $0x1,(%rax,%rdx,4)
    2f18:	4c 8d 0c 95 00 00 00 	lea    0x0(,%rdx,4),%r9
    2f1f:	00 
    2f20:	c7 44 30 04 01 00 00 	movl   $0x1,0x4(%rax,%rsi,1)
    2f27:	00 
    2f28:	83 fb 02             	cmp    $0x2,%ebx
    2f2b:	0f 84 37 02 00 00    	je     3168 <liber8tion_coding_bitmatrix+0x428>
    2f31:	c7 44 28 1c 01 00 00 	movl   $0x1,0x1c(%rax,%rbp,1)
    2f38:	00 
    2f39:	c7 44 38 1c 01 00 00 	movl   $0x1,0x1c(%rax,%rdi,1)
    2f40:	00 
    2f41:	42 c7 44 00 30 01 00 	movl   $0x1,0x30(%rax,%r8,1)
    2f48:	00 00 
    2f4a:	42 c7 44 18 18 01 00 	movl   $0x1,0x18(%rax,%r11,1)
    2f51:	00 00 
    2f53:	c7 44 30 24 01 00 00 	movl   $0x1,0x24(%rax,%rsi,1)
    2f5a:	00 
    2f5b:	42 c7 44 20 28 01 00 	movl   $0x1,0x28(%rax,%r12,1)
    2f62:	00 00 
    2f64:	c7 44 08 10 01 00 00 	movl   $0x1,0x10(%rax,%rcx,1)
    2f6b:	00 
    2f6c:	42 c7 44 08 24 01 00 	movl   $0x1,0x24(%rax,%r9,1)
    2f73:	00 00 
    2f75:	c7 44 38 20 01 00 00 	movl   $0x1,0x20(%rax,%rdi,1)
    2f7c:	00 
    2f7d:	83 fb 03             	cmp    $0x3,%ebx
    2f80:	0f 84 e2 01 00 00    	je     3168 <liber8tion_coding_bitmatrix+0x428>
    2f86:	c7 44 28 2c 01 00 00 	movl   $0x1,0x2c(%rax,%rbp,1)
    2f8d:	00 
    2f8e:	c7 44 38 48 01 00 00 	movl   $0x1,0x48(%rax,%rdi,1)
    2f95:	00 
    2f96:	42 c7 44 00 5c 01 00 	movl   $0x1,0x5c(%rax,%r8,1)
    2f9d:	00 00 
    2f9f:	42 c7 44 18 50 01 00 	movl   $0x1,0x50(%rax,%r11,1)
    2fa6:	00 00 
    2fa8:	c7 44 30 28 01 00 00 	movl   $0x1,0x28(%rax,%rsi,1)
    2faf:	00 
    2fb0:	42 c7 44 20 48 01 00 	movl   $0x1,0x48(%rax,%r12,1)
    2fb7:	00 00 
    2fb9:	c7 44 08 3c 01 00 00 	movl   $0x1,0x3c(%rax,%rcx,1)
    2fc0:	00 
    2fc1:	42 c7 44 08 34 01 00 	movl   $0x1,0x34(%rax,%r9,1)
    2fc8:	00 00 
    2fca:	42 c7 44 20 4c 01 00 	movl   $0x1,0x4c(%rax,%r12,1)
    2fd1:	00 00 
    2fd3:	83 fb 04             	cmp    $0x4,%ebx
    2fd6:	0f 84 8c 01 00 00    	je     3168 <liber8tion_coding_bitmatrix+0x428>
    2fdc:	c7 44 28 58 01 00 00 	movl   $0x1,0x58(%rax,%rbp,1)
    2fe3:	00 
    2fe4:	c7 44 38 6c 01 00 00 	movl   $0x1,0x6c(%rax,%rdi,1)
    2feb:	00 
    2fec:	42 c7 44 00 64 01 00 	movl   $0x1,0x64(%rax,%r8,1)
    2ff3:	00 00 
    2ff5:	42 c7 44 18 74 01 00 	movl   $0x1,0x74(%rax,%r11,1)
    2ffc:	00 00 
    2ffe:	c7 44 30 50 01 00 00 	movl   $0x1,0x50(%rax,%rsi,1)
    3005:	00 
    3006:	42 c7 44 20 6c 01 00 	movl   $0x1,0x6c(%rax,%r12,1)
    300d:	00 00 
    300f:	c7 44 08 58 01 00 00 	movl   $0x1,0x58(%rax,%rcx,1)
    3016:	00 
    3017:	42 c7 44 08 50 01 00 	movl   $0x1,0x50(%rax,%r9,1)
    301e:	00 00 
    3020:	42 c7 44 00 60 01 00 	movl   $0x1,0x60(%rax,%r8,1)
    3027:	00 00 
    3029:	83 fb 05             	cmp    $0x5,%ebx
    302c:	0f 84 36 01 00 00    	je     3168 <liber8tion_coding_bitmatrix+0x428>
    3032:	c7 44 28 68 01 00 00 	movl   $0x1,0x68(%rax,%rbp,1)
    3039:	00 
    303a:	c7 44 38 7c 01 00 00 	movl   $0x1,0x7c(%rax,%rdi,1)
    3041:	00 
    3042:	42 c7 84 00 8c 00 00 	movl   $0x1,0x8c(%rax,%r8,1)
    3049:	00 01 00 00 00 
    304e:	42 c7 84 18 88 00 00 	movl   $0x1,0x88(%rax,%r11,1)
    3055:	00 01 00 00 00 
    305a:	c7 44 30 7c 01 00 00 	movl   $0x1,0x7c(%rax,%rsi,1)
    3061:	00 
    3062:	42 c7 84 20 94 00 00 	movl   $0x1,0x94(%rax,%r12,1)
    3069:	00 01 00 00 00 
    306e:	c7 84 08 88 00 00 00 	movl   $0x1,0x88(%rax,%rcx,1)
    3075:	01 00 00 00 
    3079:	42 c7 44 08 70 01 00 	movl   $0x1,0x70(%rax,%r9,1)
    3080:	00 00 
    3082:	42 c7 44 08 78 01 00 	movl   $0x1,0x78(%rax,%r9,1)
    3089:	00 00 
    308b:	83 fb 06             	cmp    $0x6,%ebx
    308e:	0f 84 d4 00 00 00    	je     3168 <liber8tion_coding_bitmatrix+0x428>
    3094:	c7 84 28 90 00 00 00 	movl   $0x1,0x90(%rax,%rbp,1)
    309b:	01 00 00 00 
    309f:	c7 84 38 94 00 00 00 	movl   $0x1,0x94(%rax,%rdi,1)
    30a6:	01 00 00 00 
    30aa:	42 c7 84 00 b8 00 00 	movl   $0x1,0xb8(%rax,%r8,1)
    30b1:	00 01 00 00 00 
    30b6:	42 c7 84 18 ac 00 00 	movl   $0x1,0xac(%rax,%r11,1)
    30bd:	00 01 00 00 00 
    30c2:	c7 84 30 8c 00 00 00 	movl   $0x1,0x8c(%rax,%rsi,1)
    30c9:	01 00 00 00 
    30cd:	42 c7 84 20 b8 00 00 	movl   $0x1,0xb8(%rax,%r12,1)
    30d4:	00 01 00 00 00 
    30d9:	c7 84 08 9c 00 00 00 	movl   $0x1,0x9c(%rax,%rcx,1)
    30e0:	01 00 00 00 
    30e4:	42 c7 84 08 98 00 00 	movl   $0x1,0x98(%rax,%r9,1)
    30eb:	00 01 00 00 00 
    30f0:	c7 84 08 a0 00 00 00 	movl   $0x1,0xa0(%rax,%rcx,1)
    30f7:	01 00 00 00 
    30fb:	83 fb 07             	cmp    $0x7,%ebx
    30fe:	74 68                	je     3168 <liber8tion_coding_bitmatrix+0x428>
    3100:	c7 84 28 b4 00 00 00 	movl   $0x1,0xb4(%rax,%rbp,1)
    3107:	01 00 00 00 
    310b:	c7 84 38 d0 00 00 00 	movl   $0x1,0xd0(%rax,%rdi,1)
    3112:	01 00 00 00 
    3116:	42 c7 84 00 c4 00 00 	movl   $0x1,0xc4(%rax,%r8,1)
    311d:	00 01 00 00 00 
    3122:	42 c7 84 18 cc 00 00 	movl   $0x1,0xcc(%rax,%r11,1)
    3129:	00 01 00 00 00 
    312e:	c7 84 30 b4 00 00 00 	movl   $0x1,0xb4(%rax,%rsi,1)
    3135:	01 00 00 00 
    3139:	42 c7 84 20 c4 00 00 	movl   $0x1,0xc4(%rax,%r12,1)
    3140:	00 01 00 00 00 
    3145:	c7 84 08 ac 00 00 00 	movl   $0x1,0xac(%rax,%rcx,1)
    314c:	01 00 00 00 
    3150:	42 c7 84 08 c8 00 00 	movl   $0x1,0xc8(%rax,%r9,1)
    3157:	00 01 00 00 00 
    315c:	42 c7 84 18 bc 00 00 	movl   $0x1,0xbc(%rax,%r11,1)
    3163:	00 01 00 00 00 
    3168:	5b                   	pop    %rbx
    3169:	5d                   	pop    %rbp
    316a:	41 5c                	pop    %r12
    316c:	c3                   	retq   
    316d:	5b                   	pop    %rbx
    316e:	31 c0                	xor    %eax,%eax
    3170:	5d                   	pop    %rbp
    3171:	41 5c                	pop    %r12
    3173:	c3                   	retq   
    3174:	31 c0                	xor    %eax,%eax
    3176:	c3                   	retq   
    3177:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    317e:	00 00 

0000000000003180 <blaum_roth_coding_bitmatrix>:
    3180:	f3 0f 1e fa          	endbr64 
    3184:	39 f7                	cmp    %esi,%edi
    3186:	0f 8f db 01 00 00    	jg     3367 <blaum_roth_coding_bitmatrix+0x1e7>
    318c:	41 57                	push   %r15
    318e:	41 56                	push   %r14
    3190:	41 55                	push   %r13
    3192:	41 54                	push   %r12
    3194:	41 89 fc             	mov    %edi,%r12d
    3197:	44 0f af e6          	imul   %esi,%r12d
    319b:	55                   	push   %rbp
    319c:	89 f5                	mov    %esi,%ebp
    319e:	53                   	push   %rbx
    319f:	89 fb                	mov    %edi,%ebx
    31a1:	45 89 e6             	mov    %r12d,%r14d
    31a4:	44 0f af f6          	imul   %esi,%r14d
    31a8:	48 83 ec 18          	sub    $0x18,%rsp
    31ac:	43 8d 3c 36          	lea    (%r14,%r14,1),%edi
    31b0:	48 63 ff             	movslq %edi,%rdi
    31b3:	48 c1 e7 02          	shl    $0x2,%rdi
    31b7:	e8 34 e2 ff ff       	callq  13f0 <malloc@plt>
    31bc:	49 89 c1             	mov    %rax,%r9
    31bf:	48 85 c0             	test   %rax,%rax
    31c2:	0f 84 9a 01 00 00    	je     3362 <blaum_roth_coding_bitmatrix+0x1e2>
    31c8:	4c 63 ed             	movslq %ebp,%r13
    31cb:	48 63 c3             	movslq %ebx,%rax
    31ce:	4c 89 cf             	mov    %r9,%rdi
    31d1:	31 f6                	xor    %esi,%esi
    31d3:	4c 89 ea             	mov    %r13,%rdx
    31d6:	49 0f af d5          	imul   %r13,%rdx
    31da:	48 0f af d0          	imul   %rax,%rdx
    31de:	48 c1 e2 03          	shl    $0x3,%rdx
    31e2:	e8 59 e1 ff ff       	callq  1340 <memset@plt>
    31e7:	49 89 c1             	mov    %rax,%r9
    31ea:	85 ed                	test   %ebp,%ebp
    31ec:	7e 4c                	jle    323a <blaum_roth_coding_bitmatrix+0xba>
    31ee:	49 63 c4             	movslq %r12d,%rax
    31f1:	4c 89 cf             	mov    %r9,%rdi
    31f4:	4a 8d 0c ad 00 00 00 	lea    0x0(,%r13,4),%rcx
    31fb:	00 
    31fc:	31 f6                	xor    %esi,%esi
    31fe:	4c 8d 04 85 04 00 00 	lea    0x4(,%rax,4),%r8
    3205:	00 
    3206:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    320d:	00 00 00 
    3210:	48 89 fa             	mov    %rdi,%rdx
    3213:	31 c0                	xor    %eax,%eax
    3215:	85 db                	test   %ebx,%ebx
    3217:	7e 17                	jle    3230 <blaum_roth_coding_bitmatrix+0xb0>
    3219:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3220:	83 c0 01             	add    $0x1,%eax
    3223:	c7 02 01 00 00 00    	movl   $0x1,(%rdx)
    3229:	48 01 ca             	add    %rcx,%rdx
    322c:	39 c3                	cmp    %eax,%ebx
    322e:	75 f0                	jne    3220 <blaum_roth_coding_bitmatrix+0xa0>
    3230:	83 c6 01             	add    $0x1,%esi
    3233:	4c 01 c7             	add    %r8,%rdi
    3236:	39 f5                	cmp    %esi,%ebp
    3238:	75 d6                	jne    3210 <blaum_roth_coding_bitmatrix+0x90>
    323a:	44 8d 45 01          	lea    0x1(%rbp),%r8d
    323e:	85 db                	test   %ebx,%ebx
    3240:	0f 8e e1 00 00 00    	jle    3327 <blaum_roth_coding_bitmatrix+0x1a7>
    3246:	44 89 c0             	mov    %r8d,%eax
    3249:	89 5c 24 08          	mov    %ebx,0x8(%rsp)
    324d:	44 89 c6             	mov    %r8d,%esi
    3250:	31 ff                	xor    %edi,%edi
    3252:	c1 e8 1f             	shr    $0x1f,%eax
    3255:	44 01 c0             	add    %r8d,%eax
    3258:	d1 f8                	sar    %eax
    325a:	83 c0 01             	add    $0x1,%eax
    325d:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    3261:	4a 8d 04 ad 00 00 00 	lea    0x0(,%r13,4),%rax
    3268:	00 
    3269:	48 89 04 24          	mov    %rax,(%rsp)
    326d:	49 63 c6             	movslq %r14d,%rax
    3270:	4d 8d 2c 81          	lea    (%r9,%rax,4),%r13
    3274:	49 63 c4             	movslq %r12d,%rax
    3277:	4c 8d 3c 85 04 00 00 	lea    0x4(,%rax,4),%r15
    327e:	00 
    327f:	90                   	nop
    3280:	44 89 f2             	mov    %r14d,%edx
    3283:	85 ff                	test   %edi,%edi
    3285:	0f 84 b5 00 00 00    	je     3340 <blaum_roth_coding_bitmatrix+0x1c0>
    328b:	85 ed                	test   %ebp,%ebp
    328d:	0f 8e 7d 00 00 00    	jle    3310 <blaum_roth_coding_bitmatrix+0x190>
    3293:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    3297:	89 fb                	mov    %edi,%ebx
    3299:	d1 fb                	sar    %ebx
    329b:	01 d8                	add    %ebx,%eax
    329d:	40 f6 c7 01          	test   $0x1,%dil
    32a1:	0f 45 d8             	cmovne %eax,%ebx
    32a4:	b8 01 00 00 00       	mov    $0x1,%eax
    32a9:	83 eb 01             	sub    $0x1,%ebx
    32ac:	eb 2c                	jmp    32da <blaum_roth_coding_bitmatrix+0x15a>
    32ae:	66 90                	xchg   %ax,%ax
    32b0:	8d 0c 38             	lea    (%rax,%rdi,1),%ecx
    32b3:	41 89 c3             	mov    %eax,%r11d
    32b6:	41 29 f3             	sub    %esi,%r11d
    32b9:	44 39 c1             	cmp    %r8d,%ecx
    32bc:	41 0f 4d cb          	cmovge %r11d,%ecx
    32c0:	83 c0 01             	add    $0x1,%eax
    32c3:	8d 4c 11 ff          	lea    -0x1(%rcx,%rdx,1),%ecx
    32c7:	44 01 e2             	add    %r12d,%edx
    32ca:	48 63 c9             	movslq %ecx,%rcx
    32cd:	41 c7 04 89 01 00 00 	movl   $0x1,(%r9,%rcx,4)
    32d4:	00 
    32d5:	44 39 c0             	cmp    %r8d,%eax
    32d8:	74 36                	je     3310 <blaum_roth_coding_bitmatrix+0x190>
    32da:	39 f0                	cmp    %esi,%eax
    32dc:	75 d2                	jne    32b0 <blaum_roth_coding_bitmatrix+0x130>
    32de:	8d 0c 3a             	lea    (%rdx,%rdi,1),%ecx
    32e1:	83 c0 01             	add    $0x1,%eax
    32e4:	48 63 c9             	movslq %ecx,%rcx
    32e7:	41 c7 44 89 fc 01 00 	movl   $0x1,-0x4(%r9,%rcx,4)
    32ee:	00 00 
    32f0:	8d 0c 13             	lea    (%rbx,%rdx,1),%ecx
    32f3:	44 01 e2             	add    %r12d,%edx
    32f6:	48 63 c9             	movslq %ecx,%rcx
    32f9:	41 c7 04 89 01 00 00 	movl   $0x1,(%r9,%rcx,4)
    3300:	00 
    3301:	44 39 c0             	cmp    %r8d,%eax
    3304:	75 d4                	jne    32da <blaum_roth_coding_bitmatrix+0x15a>
    3306:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    330d:	00 00 00 
    3310:	83 c7 01             	add    $0x1,%edi
    3313:	41 01 ee             	add    %ebp,%r14d
    3316:	83 ee 01             	sub    $0x1,%esi
    3319:	4c 03 2c 24          	add    (%rsp),%r13
    331d:	39 7c 24 08          	cmp    %edi,0x8(%rsp)
    3321:	0f 85 59 ff ff ff    	jne    3280 <blaum_roth_coding_bitmatrix+0x100>
    3327:	48 83 c4 18          	add    $0x18,%rsp
    332b:	4c 89 c8             	mov    %r9,%rax
    332e:	5b                   	pop    %rbx
    332f:	5d                   	pop    %rbp
    3330:	41 5c                	pop    %r12
    3332:	41 5d                	pop    %r13
    3334:	41 5e                	pop    %r14
    3336:	41 5f                	pop    %r15
    3338:	c3                   	retq   
    3339:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3340:	85 ed                	test   %ebp,%ebp
    3342:	7e cc                	jle    3310 <blaum_roth_coding_bitmatrix+0x190>
    3344:	4c 89 ea             	mov    %r13,%rdx
    3347:	31 c0                	xor    %eax,%eax
    3349:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3350:	83 c0 01             	add    $0x1,%eax
    3353:	c7 02 01 00 00 00    	movl   $0x1,(%rdx)
    3359:	4c 01 fa             	add    %r15,%rdx
    335c:	39 c5                	cmp    %eax,%ebp
    335e:	75 f0                	jne    3350 <blaum_roth_coding_bitmatrix+0x1d0>
    3360:	eb ae                	jmp    3310 <blaum_roth_coding_bitmatrix+0x190>
    3362:	45 31 c9             	xor    %r9d,%r9d
    3365:	eb c0                	jmp    3327 <blaum_roth_coding_bitmatrix+0x1a7>
    3367:	45 31 c9             	xor    %r9d,%r9d
    336a:	4c 89 c8             	mov    %r9,%rax
    336d:	c3                   	retq   
    336e:	66 90                	xchg   %ax,%ax

0000000000003370 <jerasure_print_matrix>:
    3370:	f3 0f 1e fa          	endbr64 
    3374:	41 57                	push   %r15
    3376:	41 56                	push   %r14
    3378:	41 89 d6             	mov    %edx,%r14d
    337b:	41 55                	push   %r13
    337d:	41 54                	push   %r12
    337f:	55                   	push   %rbp
    3380:	bd 0a 00 00 00       	mov    $0xa,%ebp
    3385:	53                   	push   %rbx
    3386:	48 83 ec 48          	sub    $0x48,%rsp
    338a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
    338f:	89 74 24 04          	mov    %esi,0x4(%rsp)
    3393:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    339a:	00 00 
    339c:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    33a1:	31 c0                	xor    %eax,%eax
    33a3:	83 f9 20             	cmp    $0x20,%ecx
    33a6:	0f 85 a2 00 00 00    	jne    344e <jerasure_print_matrix+0xde>
    33ac:	8b 44 24 04          	mov    0x4(%rsp),%eax
    33b0:	85 c0                	test   %eax,%eax
    33b2:	7e 7b                	jle    342f <jerasure_print_matrix+0xbf>
    33b4:	c7 04 24 00 00 00 00 	movl   $0x0,(%rsp)
    33bb:	45 31 ff             	xor    %r15d,%r15d
    33be:	4c 8d 2d 26 73 00 00 	lea    0x7326(%rip),%r13        # a6eb <__PRETTY_FUNCTION__.5230+0x1a>
    33c5:	0f 1f 00             	nopl   (%rax)
    33c8:	45 85 f6             	test   %r14d,%r14d
    33cb:	7e 49                	jle    3416 <jerasure_print_matrix+0xa6>
    33cd:	48 63 14 24          	movslq (%rsp),%rdx
    33d1:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
    33d6:	41 8d 46 ff          	lea    -0x1(%r14),%eax
    33da:	48 01 d0             	add    %rdx,%rax
    33dd:	48 8d 1c 96          	lea    (%rsi,%rdx,4),%rbx
    33e1:	4c 8d 64 86 04       	lea    0x4(%rsi,%rax,4),%r12
    33e6:	eb 12                	jmp    33fa <jerasure_print_matrix+0x8a>
    33e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    33ef:	00 
    33f0:	bf 20 00 00 00       	mov    $0x20,%edi
    33f5:	e8 86 de ff ff       	callq  1280 <putchar@plt>
    33fa:	8b 0b                	mov    (%rbx),%ecx
    33fc:	89 ea                	mov    %ebp,%edx
    33fe:	4c 89 ee             	mov    %r13,%rsi
    3401:	bf 01 00 00 00       	mov    $0x1,%edi
    3406:	31 c0                	xor    %eax,%eax
    3408:	48 83 c3 04          	add    $0x4,%rbx
    340c:	e8 0f e0 ff ff       	callq  1420 <__printf_chk@plt>
    3411:	4c 39 e3             	cmp    %r12,%rbx
    3414:	75 da                	jne    33f0 <jerasure_print_matrix+0x80>
    3416:	bf 0a 00 00 00       	mov    $0xa,%edi
    341b:	41 83 c7 01          	add    $0x1,%r15d
    341f:	e8 5c de ff ff       	callq  1280 <putchar@plt>
    3424:	44 01 34 24          	add    %r14d,(%rsp)
    3428:	44 39 7c 24 04       	cmp    %r15d,0x4(%rsp)
    342d:	75 99                	jne    33c8 <jerasure_print_matrix+0x58>
    342f:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    3434:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    343b:	00 00 
    343d:	75 79                	jne    34b8 <jerasure_print_matrix+0x148>
    343f:	48 83 c4 48          	add    $0x48,%rsp
    3443:	5b                   	pop    %rbx
    3444:	5d                   	pop    %rbp
    3445:	41 5c                	pop    %r12
    3447:	41 5d                	pop    %r13
    3449:	41 5e                	pop    %r14
    344b:	41 5f                	pop    %r15
    344d:	c3                   	retq   
    344e:	b8 01 00 00 00       	mov    $0x1,%eax
    3453:	48 8d 5c 24 10       	lea    0x10(%rsp),%rbx
    3458:	ba 1e 00 00 00       	mov    $0x1e,%edx
    345d:	be 01 00 00 00       	mov    $0x1,%esi
    3462:	d3 e0                	shl    %cl,%eax
    3464:	48 89 df             	mov    %rbx,%rdi
    3467:	48 8d 0d 7a 72 00 00 	lea    0x727a(%rip),%rcx        # a6e8 <__PRETTY_FUNCTION__.5230+0x17>
    346e:	48 89 dd             	mov    %rbx,%rbp
    3471:	44 8d 40 ff          	lea    -0x1(%rax),%r8d
    3475:	31 c0                	xor    %eax,%eax
    3477:	e8 14 e0 ff ff       	callq  1490 <__sprintf_chk@plt>
    347c:	8b 55 00             	mov    0x0(%rbp),%edx
    347f:	48 83 c5 04          	add    $0x4,%rbp
    3483:	8d 82 ff fe fe fe    	lea    -0x1010101(%rdx),%eax
    3489:	f7 d2                	not    %edx
    348b:	21 d0                	and    %edx,%eax
    348d:	25 80 80 80 80       	and    $0x80808080,%eax
    3492:	74 e8                	je     347c <jerasure_print_matrix+0x10c>
    3494:	89 c2                	mov    %eax,%edx
    3496:	c1 ea 10             	shr    $0x10,%edx
    3499:	a9 80 80 00 00       	test   $0x8080,%eax
    349e:	0f 44 c2             	cmove  %edx,%eax
    34a1:	48 8d 55 02          	lea    0x2(%rbp),%rdx
    34a5:	48 0f 44 ea          	cmove  %rdx,%rbp
    34a9:	89 c1                	mov    %eax,%ecx
    34ab:	00 c1                	add    %al,%cl
    34ad:	48 83 dd 03          	sbb    $0x3,%rbp
    34b1:	29 dd                	sub    %ebx,%ebp
    34b3:	e9 f4 fe ff ff       	jmpq   33ac <jerasure_print_matrix+0x3c>
    34b8:	e8 43 de ff ff       	callq  1300 <__stack_chk_fail@plt>
    34bd:	0f 1f 00             	nopl   (%rax)

00000000000034c0 <jerasure_print_bitmatrix>:
    34c0:	f3 0f 1e fa          	endbr64 
    34c4:	41 57                	push   %r15
    34c6:	41 56                	push   %r14
    34c8:	41 55                	push   %r13
    34ca:	41 54                	push   %r12
    34cc:	55                   	push   %rbp
    34cd:	53                   	push   %rbx
    34ce:	48 83 ec 28          	sub    $0x28,%rsp
    34d2:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
    34d7:	89 74 24 14          	mov    %esi,0x14(%rsp)
    34db:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    34df:	85 f6                	test   %esi,%esi
    34e1:	0f 8e c1 00 00 00    	jle    35a8 <jerasure_print_bitmatrix+0xe8>
    34e7:	44 8d 62 ff          	lea    -0x1(%rdx),%r12d
    34eb:	89 cb                	mov    %ecx,%ebx
    34ed:	4c 8d 2d 43 6b 00 00 	lea    0x6b43(%rip),%r13        # a037 <_IO_stdin_used+0x37>
    34f4:	45 31 f6             	xor    %r14d,%r14d
    34f7:	c7 44 24 10 00 00 00 	movl   $0x0,0x10(%rsp)
    34fe:	00 
    34ff:	49 83 c4 01          	add    $0x1,%r12
    3503:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    3508:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    350c:	85 c0                	test   %eax,%eax
    350e:	7e 58                	jle    3568 <jerasure_print_bitmatrix+0xa8>
    3510:	48 63 44 24 10       	movslq 0x10(%rsp),%rax
    3515:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
    351a:	41 bf 01 00 00 00    	mov    $0x1,%r15d
    3520:	48 8d 2c 81          	lea    (%rcx,%rax,4),%rbp
    3524:	eb 0e                	jmp    3534 <jerasure_print_bitmatrix+0x74>
    3526:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    352d:	00 00 00 
    3530:	49 83 c7 01          	add    $0x1,%r15
    3534:	42 8b 54 bd fc       	mov    -0x4(%rbp,%r15,4),%edx
    3539:	4c 89 ee             	mov    %r13,%rsi
    353c:	bf 01 00 00 00       	mov    $0x1,%edi
    3541:	31 c0                	xor    %eax,%eax
    3543:	e8 d8 de ff ff       	callq  1420 <__printf_chk@plt>
    3548:	44 89 f8             	mov    %r15d,%eax
    354b:	4d 39 fc             	cmp    %r15,%r12
    354e:	74 18                	je     3568 <jerasure_print_bitmatrix+0xa8>
    3550:	99                   	cltd   
    3551:	f7 fb                	idiv   %ebx
    3553:	85 d2                	test   %edx,%edx
    3555:	75 d9                	jne    3530 <jerasure_print_bitmatrix+0x70>
    3557:	bf 20 00 00 00       	mov    $0x20,%edi
    355c:	e8 1f dd ff ff       	callq  1280 <putchar@plt>
    3561:	eb cd                	jmp    3530 <jerasure_print_bitmatrix+0x70>
    3563:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    3568:	bf 0a 00 00 00       	mov    $0xa,%edi
    356d:	41 83 c6 01          	add    $0x1,%r14d
    3571:	e8 0a dd ff ff       	callq  1280 <putchar@plt>
    3576:	44 39 74 24 14       	cmp    %r14d,0x14(%rsp)
    357b:	74 2b                	je     35a8 <jerasure_print_bitmatrix+0xe8>
    357d:	44 89 f0             	mov    %r14d,%eax
    3580:	99                   	cltd   
    3581:	f7 fb                	idiv   %ebx
    3583:	85 d2                	test   %edx,%edx
    3585:	74 11                	je     3598 <jerasure_print_bitmatrix+0xd8>
    3587:	8b 4c 24 0c          	mov    0xc(%rsp),%ecx
    358b:	01 4c 24 10          	add    %ecx,0x10(%rsp)
    358f:	e9 74 ff ff ff       	jmpq   3508 <jerasure_print_bitmatrix+0x48>
    3594:	0f 1f 40 00          	nopl   0x0(%rax)
    3598:	bf 0a 00 00 00       	mov    $0xa,%edi
    359d:	e8 de dc ff ff       	callq  1280 <putchar@plt>
    35a2:	eb e3                	jmp    3587 <jerasure_print_bitmatrix+0xc7>
    35a4:	0f 1f 40 00          	nopl   0x0(%rax)
    35a8:	48 83 c4 28          	add    $0x28,%rsp
    35ac:	5b                   	pop    %rbx
    35ad:	5d                   	pop    %rbp
    35ae:	41 5c                	pop    %r12
    35b0:	41 5d                	pop    %r13
    35b2:	41 5e                	pop    %r14
    35b4:	41 5f                	pop    %r15
    35b6:	c3                   	retq   
    35b7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    35be:	00 00 

00000000000035c0 <jerasure_matrix_to_bitmatrix>:
    35c0:	f3 0f 1e fa          	endbr64 
    35c4:	41 57                	push   %r15
    35c6:	41 89 d7             	mov    %edx,%r15d
    35c9:	41 56                	push   %r14
    35cb:	41 89 f6             	mov    %esi,%r14d
    35ce:	41 55                	push   %r13
    35d0:	41 54                	push   %r12
    35d2:	55                   	push   %rbp
    35d3:	48 89 cd             	mov    %rcx,%rbp
    35d6:	53                   	push   %rbx
    35d7:	89 fb                	mov    %edi,%ebx
    35d9:	48 83 ec 68          	sub    $0x68,%rsp
    35dd:	89 7c 24 40          	mov    %edi,0x40(%rsp)
    35e1:	0f af fe             	imul   %esi,%edi
    35e4:	89 74 24 58          	mov    %esi,0x58(%rsp)
    35e8:	48 89 4c 24 50       	mov    %rcx,0x50(%rsp)
    35ed:	0f af fa             	imul   %edx,%edi
    35f0:	0f af fa             	imul   %edx,%edi
    35f3:	48 63 ff             	movslq %edi,%rdi
    35f6:	48 c1 e7 02          	shl    $0x2,%rdi
    35fa:	e8 f1 dd ff ff       	callq  13f0 <malloc@plt>
    35ff:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    3604:	48 85 ed             	test   %rbp,%rbp
    3607:	0f 84 5c 01 00 00    	je     3769 <jerasure_matrix_to_bitmatrix+0x1a9>
    360d:	48 89 c7             	mov    %rax,%rdi
    3610:	89 d8                	mov    %ebx,%eax
    3612:	89 de                	mov    %ebx,%esi
    3614:	41 0f af c7          	imul   %r15d,%eax
    3618:	45 85 f6             	test   %r14d,%r14d
    361b:	0f 8e 34 01 00 00    	jle    3755 <jerasure_matrix_to_bitmatrix+0x195>
    3621:	89 c3                	mov    %eax,%ebx
    3623:	48 98                	cltq   
    3625:	48 89 7c 24 28       	mov    %rdi,0x28(%rsp)
    362a:	41 0f af df          	imul   %r15d,%ebx
    362e:	c7 44 24 34 00 00 00 	movl   $0x0,0x34(%rsp)
    3635:	00 
    3636:	c7 44 24 24 00 00 00 	movl   $0x0,0x24(%rsp)
    363d:	00 
    363e:	c7 44 24 30 00 00 00 	movl   $0x0,0x30(%rsp)
    3645:	00 
    3646:	48 63 cb             	movslq %ebx,%rcx
    3649:	89 5c 24 44          	mov    %ebx,0x44(%rsp)
    364d:	48 8d 1c 8d 00 00 00 	lea    0x0(,%rcx,4),%rbx
    3654:	00 
    3655:	49 63 cf             	movslq %r15d,%rcx
    3658:	48 89 5c 24 48       	mov    %rbx,0x48(%rsp)
    365d:	48 8d 1c 8d 00 00 00 	lea    0x0(,%rcx,4),%rbx
    3664:	00 
    3665:	48 89 5c 24 10       	mov    %rbx,0x10(%rsp)
    366a:	48 8d 1c 85 00 00 00 	lea    0x0(,%rax,4),%rbx
    3671:	00 
    3672:	8d 46 ff             	lea    -0x1(%rsi),%eax
    3675:	89 44 24 5c          	mov    %eax,0x5c(%rsp)
    3679:	8b 44 24 40          	mov    0x40(%rsp),%eax
    367d:	85 c0                	test   %eax,%eax
    367f:	0f 8e a3 00 00 00    	jle    3728 <jerasure_matrix_to_bitmatrix+0x168>
    3685:	8b 44 24 30          	mov    0x30(%rsp),%eax
    3689:	48 63 4c 24 34       	movslq 0x34(%rsp),%rcx
    368e:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
    3693:	46 8d 24 38          	lea    (%rax,%r15,1),%r12d
    3697:	8b 44 24 5c          	mov    0x5c(%rsp),%eax
    369b:	4c 8d 2c 8e          	lea    (%rsi,%rcx,4),%r13
    369f:	48 01 c8             	add    %rcx,%rax
    36a2:	48 8d 44 86 04       	lea    0x4(%rsi,%rax,4),%rax
    36a7:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    36ac:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    36b1:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    36b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    36bd:	00 00 00 
    36c0:	41 8b 7d 00          	mov    0x0(%r13),%edi
    36c4:	45 85 ff             	test   %r15d,%r15d
    36c7:	7e 47                	jle    3710 <jerasure_matrix_to_bitmatrix+0x150>
    36c9:	45 89 e6             	mov    %r12d,%r14d
    36cc:	48 8b 6c 24 08       	mov    0x8(%rsp),%rbp
    36d1:	45 29 fe             	sub    %r15d,%r14d
    36d4:	0f 1f 40 00          	nopl   0x0(%rax)
    36d8:	48 89 e8             	mov    %rbp,%rax
    36db:	31 c9                	xor    %ecx,%ecx
    36dd:	0f 1f 00             	nopl   (%rax)
    36e0:	89 fa                	mov    %edi,%edx
    36e2:	d3 fa                	sar    %cl,%edx
    36e4:	83 c1 01             	add    $0x1,%ecx
    36e7:	83 e2 01             	and    $0x1,%edx
    36ea:	89 10                	mov    %edx,(%rax)
    36ec:	48 01 d8             	add    %rbx,%rax
    36ef:	41 39 cf             	cmp    %ecx,%r15d
    36f2:	75 ec                	jne    36e0 <jerasure_matrix_to_bitmatrix+0x120>
    36f4:	44 89 fa             	mov    %r15d,%edx
    36f7:	be 02 00 00 00       	mov    $0x2,%esi
    36fc:	41 83 c6 01          	add    $0x1,%r14d
    3700:	48 83 c5 04          	add    $0x4,%rbp
    3704:	e8 07 4e 00 00       	callq  8510 <galois_single_multiply>
    3709:	89 c7                	mov    %eax,%edi
    370b:	45 39 f4             	cmp    %r14d,%r12d
    370e:	75 c8                	jne    36d8 <jerasure_matrix_to_bitmatrix+0x118>
    3710:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
    3715:	45 01 fc             	add    %r15d,%r12d
    3718:	48 01 74 24 08       	add    %rsi,0x8(%rsp)
    371d:	49 83 c5 04          	add    $0x4,%r13
    3721:	4c 3b 6c 24 18       	cmp    0x18(%rsp),%r13
    3726:	75 98                	jne    36c0 <jerasure_matrix_to_bitmatrix+0x100>
    3728:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
    372d:	83 44 24 24 01       	addl   $0x1,0x24(%rsp)
    3732:	48 01 7c 24 28       	add    %rdi,0x28(%rsp)
    3737:	8b 74 24 44          	mov    0x44(%rsp),%esi
    373b:	8b 7c 24 40          	mov    0x40(%rsp),%edi
    373f:	01 74 24 30          	add    %esi,0x30(%rsp)
    3743:	01 7c 24 34          	add    %edi,0x34(%rsp)
    3747:	8b 44 24 24          	mov    0x24(%rsp),%eax
    374b:	39 44 24 58          	cmp    %eax,0x58(%rsp)
    374f:	0f 85 24 ff ff ff    	jne    3679 <jerasure_matrix_to_bitmatrix+0xb9>
    3755:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    375a:	48 83 c4 68          	add    $0x68,%rsp
    375e:	5b                   	pop    %rbx
    375f:	5d                   	pop    %rbp
    3760:	41 5c                	pop    %r12
    3762:	41 5d                	pop    %r13
    3764:	41 5e                	pop    %r14
    3766:	41 5f                	pop    %r15
    3768:	c3                   	retq   
    3769:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
    3770:	00 00 
    3772:	eb e1                	jmp    3755 <jerasure_matrix_to_bitmatrix+0x195>
    3774:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    377b:	00 00 00 00 
    377f:	90                   	nop

0000000000003780 <jerasure_bitmatrix_dotprod>:
    3780:	f3 0f 1e fa          	endbr64 
    3784:	41 57                	push   %r15
    3786:	89 f0                	mov    %esi,%eax
    3788:	41 56                	push   %r14
    378a:	41 55                	push   %r13
    378c:	41 54                	push   %r12
    378e:	55                   	push   %rbp
    378f:	53                   	push   %rbx
    3790:	48 81 ec 88 00 00 00 	sub    $0x88,%rsp
    3797:	44 8b bc 24 d0 00 00 	mov    0xd0(%rsp),%r15d
    379e:	00 
    379f:	89 7c 24 40          	mov    %edi,0x40(%rsp)
    37a3:	48 89 54 24 30       	mov    %rdx,0x30(%rsp)
    37a8:	41 0f af c7          	imul   %r15d,%eax
    37ac:	89 74 24 2c          	mov    %esi,0x2c(%rsp)
    37b0:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    37b5:	4c 89 4c 24 50       	mov    %r9,0x50(%rsp)
    37ba:	89 c7                	mov    %eax,%edi
    37bc:	89 44 24 6c          	mov    %eax,0x6c(%rsp)
    37c0:	8b 84 24 c8 00 00 00 	mov    0xc8(%rsp),%eax
    37c7:	99                   	cltd   
    37c8:	f7 ff                	idiv   %edi
    37ca:	85 d2                	test   %edx,%edx
    37cc:	0f 85 a7 02 00 00    	jne    3a79 <jerasure_bitmatrix_dotprod+0x2f9>
    37d2:	49 63 f0             	movslq %r8d,%rsi
    37d5:	3b 74 24 40          	cmp    0x40(%rsp),%esi
    37d9:	0f 8c f3 01 00 00    	jl     39d2 <jerasure_bitmatrix_dotprod+0x252>
    37df:	89 f0                	mov    %esi,%eax
    37e1:	48 8b bc 24 c0 00 00 	mov    0xc0(%rsp),%rdi
    37e8:	00 
    37e9:	2b 44 24 40          	sub    0x40(%rsp),%eax
    37ed:	48 98                	cltq   
    37ef:	8b b4 24 c8 00 00 00 	mov    0xc8(%rsp),%esi
    37f6:	48 8b 04 c7          	mov    (%rdi,%rax,8),%rax
    37fa:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
    37ff:	85 f6                	test   %esi,%esi
    3801:	0f 8e e8 01 00 00    	jle    39ef <jerasure_bitmatrix_dotprod+0x26f>
    3807:	48 63 44 24 6c       	movslq 0x6c(%rsp),%rax
    380c:	c7 44 24 68 00 00 00 	movl   $0x0,0x68(%rsp)
    3813:	00 
    3814:	45 89 fe             	mov    %r15d,%r14d
    3817:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    381e:	00 00 
    3820:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    3825:	8b 44 24 40          	mov    0x40(%rsp),%eax
    3829:	83 e8 01             	sub    $0x1,%eax
    382c:	89 44 24 60          	mov    %eax,0x60(%rsp)
    3830:	0f af 44 24 2c       	imul   0x2c(%rsp),%eax
    3835:	89 44 24 64          	mov    %eax,0x64(%rsp)
    3839:	8b 44 24 2c          	mov    0x2c(%rsp),%eax
    383d:	85 c0                	test   %eax,%eax
    383f:	0f 8e 0c 02 00 00    	jle    3a51 <jerasure_bitmatrix_dotprod+0x2d1>
    3845:	49 63 c6             	movslq %r14d,%rax
    3848:	c7 44 24 5c 00 00 00 	movl   $0x0,0x5c(%rsp)
    384f:	00 
    3850:	4c 8b 7c 24 70       	mov    0x70(%rsp),%r15
    3855:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    385a:	4c 03 7c 24 08       	add    0x8(%rsp),%r15
    385f:	c7 44 24 58 00 00 00 	movl   $0x0,0x58(%rsp)
    3866:	00 
    3867:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    386e:	00 00 
    3870:	8b 4c 24 40          	mov    0x40(%rsp),%ecx
    3874:	85 c9                	test   %ecx,%ecx
    3876:	0f 8e bd 01 00 00    	jle    3a39 <jerasure_bitmatrix_dotprod+0x2b9>
    387c:	8b 44 24 60          	mov    0x60(%rsp),%eax
    3880:	8b 7c 24 2c          	mov    0x2c(%rsp),%edi
    3884:	4d 89 fd             	mov    %r15,%r13
    3887:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    388e:	00 00 
    3890:	8d 4f ff             	lea    -0x1(%rdi),%ecx
    3893:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    3898:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    389d:	8b 44 24 5c          	mov    0x5c(%rsp),%eax
    38a1:	89 4c 24 44          	mov    %ecx,0x44(%rsp)
    38a5:	48 8d 4f 04          	lea    0x4(%rdi),%rcx
    38a9:	89 44 24 28          	mov    %eax,0x28(%rsp)
    38ad:	31 c0                	xor    %eax,%eax
    38af:	48 83 7c 24 20 00    	cmpq   $0x0,0x20(%rsp)
    38b5:	48 89 4c 24 48       	mov    %rcx,0x48(%rsp)
    38ba:	41 89 c7             	mov    %eax,%r15d
    38bd:	0f 84 f8 00 00 00    	je     39bb <jerasure_bitmatrix_dotprod+0x23b>
    38c3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    38c8:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    38cd:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    38d2:	48 63 14 b8          	movslq (%rax,%rdi,4),%rdx
    38d6:	3b 54 24 40          	cmp    0x40(%rsp),%edx
    38da:	0f 8d 28 01 00 00    	jge    3a08 <jerasure_bitmatrix_dotprod+0x288>
    38e0:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    38e5:	48 8b 04 d0          	mov    (%rax,%rdx,8),%rax
    38e9:	48 89 04 24          	mov    %rax,(%rsp)
    38ed:	48 63 4c 24 28       	movslq 0x28(%rsp),%rcx
    38f2:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
    38f7:	31 ed                	xor    %ebp,%ebp
    38f9:	8b 54 24 44          	mov    0x44(%rsp),%edx
    38fd:	48 8d 1c 88          	lea    (%rax,%rcx,4),%rbx
    3901:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
    3906:	48 01 ca             	add    %rcx,%rdx
    3909:	4c 8d 24 90          	lea    (%rax,%rdx,4),%r12
    390d:	eb 39                	jmp    3948 <jerasure_bitmatrix_dotprod+0x1c8>
    390f:	90                   	nop
    3910:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
    3915:	4c 89 ef             	mov    %r13,%rdi
    3918:	41 bf 01 00 00 00    	mov    $0x1,%r15d
    391e:	e8 8d da ff ff       	callq  13b0 <memcpy@plt>
    3923:	66 0f ef c0          	pxor   %xmm0,%xmm0
    3927:	f2 41 0f 2a c6       	cvtsi2sd %r14d,%xmm0
    392c:	f2 0f 58 05 1c d8 00 	addsd  0xd81c(%rip),%xmm0        # 11150 <jerasure_total_memcpy_bytes>
    3933:	00 
    3934:	f2 0f 11 05 14 d8 00 	movsd  %xmm0,0xd814(%rip)        # 11150 <jerasure_total_memcpy_bytes>
    393b:	00 
    393c:	48 83 c3 04          	add    $0x4,%rbx
    3940:	44 01 f5             	add    %r14d,%ebp
    3943:	4c 39 e3             	cmp    %r12,%rbx
    3946:	74 4a                	je     3992 <jerasure_bitmatrix_dotprod+0x212>
    3948:	8b 13                	mov    (%rbx),%edx
    394a:	85 d2                	test   %edx,%edx
    394c:	74 ee                	je     393c <jerasure_bitmatrix_dotprod+0x1bc>
    394e:	48 63 f5             	movslq %ebp,%rsi
    3951:	48 03 74 24 08       	add    0x8(%rsp),%rsi
    3956:	48 03 34 24          	add    (%rsp),%rsi
    395a:	45 85 ff             	test   %r15d,%r15d
    395d:	74 b1                	je     3910 <jerasure_bitmatrix_dotprod+0x190>
    395f:	44 89 f1             	mov    %r14d,%ecx
    3962:	4c 89 ea             	mov    %r13,%rdx
    3965:	4c 89 ef             	mov    %r13,%rdi
    3968:	48 83 c3 04          	add    $0x4,%rbx
    396c:	e8 6f 46 00 00       	callq  7fe0 <galois_region_xor>
    3971:	66 0f ef c0          	pxor   %xmm0,%xmm0
    3975:	44 01 f5             	add    %r14d,%ebp
    3978:	f2 41 0f 2a c6       	cvtsi2sd %r14d,%xmm0
    397d:	f2 0f 58 05 db d7 00 	addsd  0xd7db(%rip),%xmm0        # 11160 <jerasure_total_xor_bytes>
    3984:	00 
    3985:	f2 0f 11 05 d3 d7 00 	movsd  %xmm0,0xd7d3(%rip)        # 11160 <jerasure_total_xor_bytes>
    398c:	00 
    398d:	4c 39 e3             	cmp    %r12,%rbx
    3990:	75 b6                	jne    3948 <jerasure_bitmatrix_dotprod+0x1c8>
    3992:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    3997:	8b 7c 24 2c          	mov    0x2c(%rsp),%edi
    399b:	01 7c 24 28          	add    %edi,0x28(%rsp)
    399f:	48 8d 50 01          	lea    0x1(%rax),%rdx
    39a3:	48 3b 44 24 38       	cmp    0x38(%rsp),%rax
    39a8:	74 7e                	je     3a28 <jerasure_bitmatrix_dotprod+0x2a8>
    39aa:	48 83 7c 24 20 00    	cmpq   $0x0,0x20(%rsp)
    39b0:	48 89 54 24 10       	mov    %rdx,0x10(%rsp)
    39b5:	0f 85 0d ff ff ff    	jne    38c8 <jerasure_bitmatrix_dotprod+0x148>
    39bb:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    39c0:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    39c5:	48 8b 04 c8          	mov    (%rax,%rcx,8),%rax
    39c9:	48 89 04 24          	mov    %rax,(%rsp)
    39cd:	e9 1b ff ff ff       	jmpq   38ed <jerasure_bitmatrix_dotprod+0x16d>
    39d2:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    39d7:	48 8b 04 f0          	mov    (%rax,%rsi,8),%rax
    39db:	8b b4 24 c8 00 00 00 	mov    0xc8(%rsp),%esi
    39e2:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
    39e7:	85 f6                	test   %esi,%esi
    39e9:	0f 8f 18 fe ff ff    	jg     3807 <jerasure_bitmatrix_dotprod+0x87>
    39ef:	48 81 c4 88 00 00 00 	add    $0x88,%rsp
    39f6:	5b                   	pop    %rbx
    39f7:	5d                   	pop    %rbp
    39f8:	41 5c                	pop    %r12
    39fa:	41 5d                	pop    %r13
    39fc:	41 5e                	pop    %r14
    39fe:	41 5f                	pop    %r15
    3a00:	c3                   	retq   
    3a01:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3a08:	48 8b 84 24 c0 00 00 	mov    0xc0(%rsp),%rax
    3a0f:	00 
    3a10:	2b 54 24 40          	sub    0x40(%rsp),%edx
    3a14:	48 63 d2             	movslq %edx,%rdx
    3a17:	48 8b 04 d0          	mov    (%rax,%rdx,8),%rax
    3a1b:	48 89 04 24          	mov    %rax,(%rsp)
    3a1f:	e9 c9 fe ff ff       	jmpq   38ed <jerasure_bitmatrix_dotprod+0x16d>
    3a24:	0f 1f 40 00          	nopl   0x0(%rax)
    3a28:	8b 44 24 5c          	mov    0x5c(%rsp),%eax
    3a2c:	4d 89 ef             	mov    %r13,%r15
    3a2f:	01 f8                	add    %edi,%eax
    3a31:	03 44 24 64          	add    0x64(%rsp),%eax
    3a35:	89 44 24 5c          	mov    %eax,0x5c(%rsp)
    3a39:	83 44 24 58 01       	addl   $0x1,0x58(%rsp)
    3a3e:	4c 03 7c 24 18       	add    0x18(%rsp),%r15
    3a43:	8b 44 24 58          	mov    0x58(%rsp),%eax
    3a47:	39 44 24 2c          	cmp    %eax,0x2c(%rsp)
    3a4b:	0f 85 1f fe ff ff    	jne    3870 <jerasure_bitmatrix_dotprod+0xf0>
    3a51:	8b 7c 24 6c          	mov    0x6c(%rsp),%edi
    3a55:	48 8b 4c 24 78       	mov    0x78(%rsp),%rcx
    3a5a:	01 7c 24 68          	add    %edi,0x68(%rsp)
    3a5e:	8b 44 24 68          	mov    0x68(%rsp),%eax
    3a62:	48 01 4c 24 08       	add    %rcx,0x8(%rsp)
    3a67:	39 84 24 c8 00 00 00 	cmp    %eax,0xc8(%rsp)
    3a6e:	0f 8f c5 fd ff ff    	jg     3839 <jerasure_bitmatrix_dotprod+0xb9>
    3a74:	e9 76 ff ff ff       	jmpq   39ef <jerasure_bitmatrix_dotprod+0x26f>
    3a79:	48 8b 3d c0 d6 00 00 	mov    0xd6c0(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    3a80:	b9 25 00 00 00       	mov    $0x25,%ecx
    3a85:	48 8d 15 64 6c 00 00 	lea    0x6c64(%rip),%rdx        # a6f0 <__PRETTY_FUNCTION__.5230+0x1f>
    3a8c:	31 c0                	xor    %eax,%eax
    3a8e:	be 01 00 00 00       	mov    $0x1,%esi
    3a93:	e8 d8 d9 ff ff       	callq  1470 <__fprintf_chk@plt>
    3a98:	bf 01 00 00 00       	mov    $0x1,%edi
    3a9d:	e8 ae d9 ff ff       	callq  1450 <exit@plt>
    3aa2:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    3aa9:	00 00 00 00 
    3aad:	0f 1f 00             	nopl   (%rax)

0000000000003ab0 <jerasure_do_parity>:
    3ab0:	f3 0f 1e fa          	endbr64 
    3ab4:	41 56                	push   %r14
    3ab6:	41 89 fe             	mov    %edi,%r14d
    3ab9:	41 55                	push   %r13
    3abb:	49 89 f5             	mov    %rsi,%r13
    3abe:	41 54                	push   %r12
    3ac0:	55                   	push   %rbp
    3ac1:	48 89 d5             	mov    %rdx,%rbp
    3ac4:	48 63 d1             	movslq %ecx,%rdx
    3ac7:	53                   	push   %rbx
    3ac8:	48 89 ef             	mov    %rbp,%rdi
    3acb:	49 89 d4             	mov    %rdx,%r12
    3ace:	48 83 ec 10          	sub    $0x10,%rsp
    3ad2:	48 8b 36             	mov    (%rsi),%rsi
    3ad5:	e8 d6 d8 ff ff       	callq  13b0 <memcpy@plt>
    3ada:	66 0f ef c9          	pxor   %xmm1,%xmm1
    3ade:	f2 0f 10 05 6a d6 00 	movsd  0xd66a(%rip),%xmm0        # 11150 <jerasure_total_memcpy_bytes>
    3ae5:	00 
    3ae6:	f2 41 0f 2a cc       	cvtsi2sd %r12d,%xmm1
    3aeb:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    3aef:	f2 0f 11 4c 24 08    	movsd  %xmm1,0x8(%rsp)
    3af5:	f2 0f 11 05 53 d6 00 	movsd  %xmm0,0xd653(%rip)        # 11150 <jerasure_total_memcpy_bytes>
    3afc:	00 
    3afd:	41 83 fe 01          	cmp    $0x1,%r14d
    3b01:	7e 3d                	jle    3b40 <jerasure_do_parity+0x90>
    3b03:	41 8d 46 fe          	lea    -0x2(%r14),%eax
    3b07:	49 8d 5d 08          	lea    0x8(%r13),%rbx
    3b0b:	4d 8d 6c c5 10       	lea    0x10(%r13,%rax,8),%r13
    3b10:	48 8b 3b             	mov    (%rbx),%rdi
    3b13:	44 89 e1             	mov    %r12d,%ecx
    3b16:	48 89 ea             	mov    %rbp,%rdx
    3b19:	48 89 ee             	mov    %rbp,%rsi
    3b1c:	48 83 c3 08          	add    $0x8,%rbx
    3b20:	e8 bb 44 00 00       	callq  7fe0 <galois_region_xor>
    3b25:	f2 0f 10 44 24 08    	movsd  0x8(%rsp),%xmm0
    3b2b:	f2 0f 58 05 2d d6 00 	addsd  0xd62d(%rip),%xmm0        # 11160 <jerasure_total_xor_bytes>
    3b32:	00 
    3b33:	f2 0f 11 05 25 d6 00 	movsd  %xmm0,0xd625(%rip)        # 11160 <jerasure_total_xor_bytes>
    3b3a:	00 
    3b3b:	4c 39 eb             	cmp    %r13,%rbx
    3b3e:	75 d0                	jne    3b10 <jerasure_do_parity+0x60>
    3b40:	48 83 c4 10          	add    $0x10,%rsp
    3b44:	5b                   	pop    %rbx
    3b45:	5d                   	pop    %rbp
    3b46:	41 5c                	pop    %r12
    3b48:	41 5d                	pop    %r13
    3b4a:	41 5e                	pop    %r14
    3b4c:	c3                   	retq   
    3b4d:	0f 1f 00             	nopl   (%rax)

0000000000003b50 <jerasure_invert_matrix>:
    3b50:	f3 0f 1e fa          	endbr64 
    3b54:	41 57                	push   %r15
    3b56:	41 56                	push   %r14
    3b58:	41 55                	push   %r13
    3b5a:	41 54                	push   %r12
    3b5c:	55                   	push   %rbp
    3b5d:	53                   	push   %rbx
    3b5e:	48 89 f3             	mov    %rsi,%rbx
    3b61:	8d 72 ff             	lea    -0x1(%rdx),%esi
    3b64:	48 81 ec a8 00 00 00 	sub    $0xa8,%rsp
    3b6b:	89 54 24 30          	mov    %edx,0x30(%rsp)
    3b6f:	c7 44 24 38 00 00 00 	movl   $0x0,0x38(%rsp)
    3b76:	00 
    3b77:	89 b4 24 9c 00 00 00 	mov    %esi,0x9c(%rsp)
    3b7e:	85 d2                	test   %edx,%edx
    3b80:	0f 8e da 04 00 00    	jle    4060 <jerasure_invert_matrix+0x510>
    3b86:	44 8b 64 24 30       	mov    0x30(%rsp),%r12d
    3b8b:	48 89 fd             	mov    %rdi,%rbp
    3b8e:	41 89 cf             	mov    %ecx,%r15d
    3b91:	31 ff                	xor    %edi,%edi
    3b93:	31 c9                	xor    %ecx,%ecx
    3b95:	0f 1f 00             	nopl   (%rax)
    3b98:	48 63 c7             	movslq %edi,%rax
    3b9b:	4c 8d 1c 83          	lea    (%rbx,%rax,4),%r11
    3b9f:	31 c0                	xor    %eax,%eax
    3ba1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3ba8:	31 d2                	xor    %edx,%edx
    3baa:	39 c1                	cmp    %eax,%ecx
    3bac:	0f 94 c2             	sete   %dl
    3baf:	41 89 14 83          	mov    %edx,(%r11,%rax,4)
    3bb3:	48 89 c2             	mov    %rax,%rdx
    3bb6:	48 83 c0 01          	add    $0x1,%rax
    3bba:	48 39 f2             	cmp    %rsi,%rdx
    3bbd:	75 e9                	jne    3ba8 <jerasure_invert_matrix+0x58>
    3bbf:	8d 41 01             	lea    0x1(%rcx),%eax
    3bc2:	44 01 e7             	add    %r12d,%edi
    3bc5:	41 39 c4             	cmp    %eax,%r12d
    3bc8:	74 04                	je     3bce <jerasure_invert_matrix+0x7e>
    3bca:	89 c1                	mov    %eax,%ecx
    3bcc:	eb ca                	jmp    3b98 <jerasure_invert_matrix+0x48>
    3bce:	89 84 24 98 00 00 00 	mov    %eax,0x98(%rsp)
    3bd5:	48 63 44 24 30       	movslq 0x30(%rsp),%rax
    3bda:	89 4c 24 38          	mov    %ecx,0x38(%rsp)
    3bde:	48 8d 3c 85 04 00 00 	lea    0x4(,%rax,4),%rdi
    3be5:	00 
    3be6:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    3beb:	48 c1 e0 02          	shl    $0x2,%rax
    3bef:	48 89 bc 24 90 00 00 	mov    %rdi,0x90(%rsp)
    3bf6:	00 
    3bf7:	48 8d 7d 04          	lea    0x4(%rbp),%rdi
    3bfb:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    3c00:	89 c8                	mov    %ecx,%eax
    3c02:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    3c07:	48 89 bc 24 88 00 00 	mov    %rdi,0x88(%rsp)
    3c0e:	00 
    3c0f:	48 8d 3c 87          	lea    (%rdi,%rax,4),%rdi
    3c13:	48 83 c0 01          	add    $0x1,%rax
    3c17:	48 89 bc 24 80 00 00 	mov    %rdi,0x80(%rsp)
    3c1e:	00 
    3c1f:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
    3c24:	48 89 6c 24 60       	mov    %rbp,0x60(%rsp)
    3c29:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
    3c30:	00 00 
    3c32:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
    3c39:	00 00 
    3c3b:	c7 44 24 6c 00 00 00 	movl   $0x0,0x6c(%rsp)
    3c42:	00 
    3c43:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
    3c4a:	00 00 
    3c4c:	0f 1f 40 00          	nopl   0x0(%rax)
    3c50:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    3c55:	8b 30                	mov    (%rax),%esi
    3c57:	8b 44 24 48          	mov    0x48(%rsp),%eax
    3c5b:	83 c0 01             	add    $0x1,%eax
    3c5e:	89 44 24 18          	mov    %eax,0x18(%rsp)
    3c62:	85 f6                	test   %esi,%esi
    3c64:	0f 84 96 01 00 00    	je     3e00 <jerasure_invert_matrix+0x2b0>
    3c6a:	83 fe 01             	cmp    $0x1,%esi
    3c6d:	0f 85 36 02 00 00    	jne    3ea9 <jerasure_invert_matrix+0x359>
    3c73:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    3c78:	48 39 7c 24 48       	cmp    %rdi,0x48(%rsp)
    3c7d:	0f 84 a4 02 00 00    	je     3f27 <jerasure_invert_matrix+0x3d7>
    3c83:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    3c88:	48 03 44 24 28       	add    0x28(%rsp),%rax
    3c8d:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    3c92:	8b 7c 24 30          	mov    0x30(%rsp),%edi
    3c96:	01 7c 24 6c          	add    %edi,0x6c(%rsp)
    3c9a:	8b 44 24 6c          	mov    0x6c(%rsp),%eax
    3c9e:	89 44 24 20          	mov    %eax,0x20(%rsp)
    3ca2:	48 89 d8             	mov    %rbx,%rax
    3ca5:	48 89 eb             	mov    %rbp,%rbx
    3ca8:	48 89 c5             	mov    %rax,%rbp
    3cab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    3cb0:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    3cb5:	44 8b 30             	mov    (%rax),%r14d
    3cb8:	45 85 f6             	test   %r14d,%r14d
    3cbb:	74 6a                	je     3d27 <jerasure_invert_matrix+0x1d7>
    3cbd:	41 83 fe 01          	cmp    $0x1,%r14d
    3cc1:	0f 84 89 00 00 00    	je     3d50 <jerasure_invert_matrix+0x200>
    3cc7:	48 63 44 24 20       	movslq 0x20(%rsp),%rax
    3ccc:	4c 8b 6c 24 50       	mov    0x50(%rsp),%r13
    3cd1:	4c 8d 24 85 00 00 00 	lea    0x0(,%rax,4),%r12
    3cd8:	00 
    3cd9:	48 03 44 24 70       	add    0x70(%rsp),%rax
    3cde:	48 c1 e0 02          	shl    $0x2,%rax
    3ce2:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    3ce7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    3cee:	00 00 
    3cf0:	42 8b 34 2b          	mov    (%rbx,%r13,1),%esi
    3cf4:	44 89 fa             	mov    %r15d,%edx
    3cf7:	44 89 f7             	mov    %r14d,%edi
    3cfa:	e8 11 48 00 00       	callq  8510 <galois_single_multiply>
    3cff:	42 31 04 23          	xor    %eax,(%rbx,%r12,1)
    3d03:	44 89 fa             	mov    %r15d,%edx
    3d06:	44 89 f7             	mov    %r14d,%edi
    3d09:	42 8b 74 2d 00       	mov    0x0(%rbp,%r13,1),%esi
    3d0e:	49 83 c5 04          	add    $0x4,%r13
    3d12:	e8 f9 47 00 00       	callq  8510 <galois_single_multiply>
    3d17:	42 31 44 25 00       	xor    %eax,0x0(%rbp,%r12,1)
    3d1c:	49 83 c4 04          	add    $0x4,%r12
    3d20:	4c 3b 64 24 08       	cmp    0x8(%rsp),%r12
    3d25:	75 c9                	jne    3cf0 <jerasure_invert_matrix+0x1a0>
    3d27:	8b 4c 24 18          	mov    0x18(%rsp),%ecx
    3d2b:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    3d30:	48 01 54 24 10       	add    %rdx,0x10(%rsp)
    3d35:	8b 54 24 30          	mov    0x30(%rsp),%edx
    3d39:	8d 41 01             	lea    0x1(%rcx),%eax
    3d3c:	01 54 24 20          	add    %edx,0x20(%rsp)
    3d40:	3b 4c 24 38          	cmp    0x38(%rsp),%ecx
    3d44:	74 62                	je     3da8 <jerasure_invert_matrix+0x258>
    3d46:	89 44 24 18          	mov    %eax,0x18(%rsp)
    3d4a:	e9 61 ff ff ff       	jmpq   3cb0 <jerasure_invert_matrix+0x160>
    3d4f:	90                   	nop
    3d50:	48 63 74 24 20       	movslq 0x20(%rsp),%rsi
    3d55:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    3d5a:	48 8b 8c 24 88 00 00 	mov    0x88(%rsp),%rcx
    3d61:	00 
    3d62:	48 01 f7             	add    %rsi,%rdi
    3d65:	48 8d 14 b5 00 00 00 	lea    0x0(,%rsi,4),%rdx
    3d6c:	00 
    3d6d:	48 8d 3c b9          	lea    (%rcx,%rdi,4),%rdi
    3d71:	48 8b 4c 24 78       	mov    0x78(%rsp),%rcx
    3d76:	48 8d 04 13          	lea    (%rbx,%rdx,1),%rax
    3d7a:	48 01 ea             	add    %rbp,%rdx
    3d7d:	48 29 f1             	sub    %rsi,%rcx
    3d80:	48 89 ce             	mov    %rcx,%rsi
    3d83:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    3d88:	8b 0c b0             	mov    (%rax,%rsi,4),%ecx
    3d8b:	31 08                	xor    %ecx,(%rax)
    3d8d:	48 83 c0 04          	add    $0x4,%rax
    3d91:	8b 0c b2             	mov    (%rdx,%rsi,4),%ecx
    3d94:	31 0a                	xor    %ecx,(%rdx)
    3d96:	48 83 c2 04          	add    $0x4,%rdx
    3d9a:	48 39 f8             	cmp    %rdi,%rax
    3d9d:	75 e9                	jne    3d88 <jerasure_invert_matrix+0x238>
    3d9f:	e9 83 ff ff ff       	jmpq   3d27 <jerasure_invert_matrix+0x1d7>
    3da4:	0f 1f 40 00          	nopl   0x0(%rax)
    3da8:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
    3dad:	48 89 e8             	mov    %rbp,%rax
    3db0:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    3db5:	48 89 dd             	mov    %rbx,%rbp
    3db8:	48 8b 94 24 90 00 00 	mov    0x90(%rsp),%rdx
    3dbf:	00 
    3dc0:	48 89 c3             	mov    %rax,%rbx
    3dc3:	48 01 54 24 60       	add    %rdx,0x60(%rsp)
    3dc8:	48 01 4c 24 78       	add    %rcx,0x78(%rsp)
    3dcd:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    3dd2:	48 8d 47 01          	lea    0x1(%rdi),%rax
    3dd6:	48 01 54 24 50       	add    %rdx,0x50(%rsp)
    3ddb:	48 01 94 24 80 00 00 	add    %rdx,0x80(%rsp)
    3de2:	00 
    3de3:	48 3b 7c 24 58       	cmp    0x58(%rsp),%rdi
    3de8:	0f 84 39 01 00 00    	je     3f27 <jerasure_invert_matrix+0x3d7>
    3dee:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    3df3:	e9 58 fe ff ff       	jmpq   3c50 <jerasure_invert_matrix+0x100>
    3df8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    3dff:	00 
    3e00:	89 c7                	mov    %eax,%edi
    3e02:	39 84 24 98 00 00 00 	cmp    %eax,0x98(%rsp)
    3e09:	0f 8e 55 02 00 00    	jle    4064 <jerasure_invert_matrix+0x514>
    3e0f:	8b 44 24 6c          	mov    0x6c(%rsp),%eax
    3e13:	03 44 24 30          	add    0x30(%rsp),%eax
    3e17:	48 98                	cltq   
    3e19:	48 03 44 24 48       	add    0x48(%rsp),%rax
    3e1e:	8b 74 24 38          	mov    0x38(%rsp),%esi
    3e22:	48 8d 54 85 00       	lea    0x0(%rbp,%rax,4),%rdx
    3e27:	89 f8                	mov    %edi,%eax
    3e29:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    3e2e:	eb 10                	jmp    3e40 <jerasure_invert_matrix+0x2f0>
    3e30:	8d 48 01             	lea    0x1(%rax),%ecx
    3e33:	48 01 fa             	add    %rdi,%rdx
    3e36:	39 f0                	cmp    %esi,%eax
    3e38:	0f 84 d1 01 00 00    	je     400f <jerasure_invert_matrix+0x4bf>
    3e3e:	89 c8                	mov    %ecx,%eax
    3e40:	8b 0a                	mov    (%rdx),%ecx
    3e42:	85 c9                	test   %ecx,%ecx
    3e44:	74 ea                	je     3e30 <jerasure_invert_matrix+0x2e0>
    3e46:	8b bc 24 98 00 00 00 	mov    0x98(%rsp),%edi
    3e4d:	39 c7                	cmp    %eax,%edi
    3e4f:	0f 84 ba 01 00 00    	je     400f <jerasure_invert_matrix+0x4bf>
    3e55:	0f af c7             	imul   %edi,%eax
    3e58:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
    3e5d:	4c 8b 9c 24 80 00 00 	mov    0x80(%rsp),%r11
    3e64:	00 
    3e65:	48 8d 54 3d 00       	lea    0x0(%rbp,%rdi,1),%rdx
    3e6a:	48 8d 0c 3b          	lea    (%rbx,%rdi,1),%rcx
    3e6e:	48 98                	cltq   
    3e70:	48 2b 44 24 78       	sub    0x78(%rsp),%rax
    3e75:	0f 1f 00             	nopl   (%rax)
    3e78:	8b 32                	mov    (%rdx),%esi
    3e7a:	8b 3c 82             	mov    (%rdx,%rax,4),%edi
    3e7d:	89 3a                	mov    %edi,(%rdx)
    3e7f:	89 34 82             	mov    %esi,(%rdx,%rax,4)
    3e82:	8b 31                	mov    (%rcx),%esi
    3e84:	48 83 c2 04          	add    $0x4,%rdx
    3e88:	8b 3c 81             	mov    (%rcx,%rax,4),%edi
    3e8b:	89 39                	mov    %edi,(%rcx)
    3e8d:	89 34 81             	mov    %esi,(%rcx,%rax,4)
    3e90:	48 83 c1 04          	add    $0x4,%rcx
    3e94:	4c 39 da             	cmp    %r11,%rdx
    3e97:	75 df                	jne    3e78 <jerasure_invert_matrix+0x328>
    3e99:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    3e9e:	8b 30                	mov    (%rax),%esi
    3ea0:	83 fe 01             	cmp    $0x1,%esi
    3ea3:	0f 84 ca fd ff ff    	je     3c73 <jerasure_invert_matrix+0x123>
    3ea9:	44 89 fa             	mov    %r15d,%edx
    3eac:	bf 01 00 00 00       	mov    $0x1,%edi
    3eb1:	e8 7a 46 00 00       	callq  8530 <galois_single_divide>
    3eb6:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
    3ebb:	41 89 c5             	mov    %eax,%r13d
    3ebe:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    3ec3:	4c 8d 74 05 00       	lea    0x0(%rbp,%rax,1),%r14
    3ec8:	4c 8d 24 03          	lea    (%rbx,%rax,1),%r12
    3ecc:	4c 89 f3             	mov    %r14,%rbx
    3ecf:	4c 8b b4 24 80 00 00 	mov    0x80(%rsp),%r14
    3ed6:	00 
    3ed7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    3ede:	00 00 
    3ee0:	8b 3b                	mov    (%rbx),%edi
    3ee2:	44 89 fa             	mov    %r15d,%edx
    3ee5:	44 89 ee             	mov    %r13d,%esi
    3ee8:	48 83 c3 04          	add    $0x4,%rbx
    3eec:	49 83 c4 04          	add    $0x4,%r12
    3ef0:	e8 1b 46 00 00       	callq  8510 <galois_single_multiply>
    3ef5:	44 89 fa             	mov    %r15d,%edx
    3ef8:	44 89 ee             	mov    %r13d,%esi
    3efb:	89 43 fc             	mov    %eax,-0x4(%rbx)
    3efe:	41 8b 7c 24 fc       	mov    -0x4(%r12),%edi
    3f03:	e8 08 46 00 00       	callq  8510 <galois_single_multiply>
    3f08:	41 89 44 24 fc       	mov    %eax,-0x4(%r12)
    3f0d:	4c 39 f3             	cmp    %r14,%rbx
    3f10:	75 ce                	jne    3ee0 <jerasure_invert_matrix+0x390>
    3f12:	48 8b 5c 24 08       	mov    0x8(%rsp),%rbx
    3f17:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    3f1c:	48 39 7c 24 48       	cmp    %rdi,0x48(%rsp)
    3f21:	0f 85 5c fd ff ff    	jne    3c83 <jerasure_invert_matrix+0x133>
    3f27:	48 63 44 24 38       	movslq 0x38(%rsp),%rax
    3f2c:	8b 7c 24 30          	mov    0x30(%rsp),%edi
    3f30:	89 44 24 10          	mov    %eax,0x10(%rsp)
    3f34:	48 89 c1             	mov    %rax,%rcx
    3f37:	48 8d 44 85 00       	lea    0x0(%rbp,%rax,4),%rax
    3f3c:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    3f41:	89 f8                	mov    %edi,%eax
    3f43:	0f af f9             	imul   %ecx,%edi
    3f46:	f7 d8                	neg    %eax
    3f48:	48 98                	cltq   
    3f4a:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    3f4f:	48 63 c7             	movslq %edi,%rax
    3f52:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    3f57:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    3f5c:	48 c1 e0 02          	shl    $0x2,%rax
    3f60:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    3f65:	8b 84 24 9c 00 00 00 	mov    0x9c(%rsp),%eax
    3f6c:	48 8d 5c 83 04       	lea    0x4(%rbx,%rax,4),%rbx
    3f71:	48 f7 d0             	not    %rax
    3f74:	48 c1 e0 02          	shl    $0x2,%rax
    3f78:	48 89 5c 24 48       	mov    %rbx,0x48(%rsp)
    3f7d:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    3f82:	8b 44 24 10          	mov    0x10(%rsp),%eax
    3f86:	85 c0                	test   %eax,%eax
    3f88:	0f 8e c7 00 00 00    	jle    4055 <jerasure_invert_matrix+0x505>
    3f8e:	c7 44 24 08 00 00 00 	movl   $0x0,0x8(%rsp)
    3f95:	00 
    3f96:	48 8b 5c 24 48       	mov    0x48(%rsp),%rbx
    3f9b:	4c 8b 74 24 28       	mov    0x28(%rsp),%r14
    3fa0:	48 8b 6c 24 38       	mov    0x38(%rsp),%rbp
    3fa5:	eb 28                	jmp    3fcf <jerasure_invert_matrix+0x47f>
    3fa7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    3fae:	00 00 
    3fb0:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
    3fb5:	83 44 24 08 01       	addl   $0x1,0x8(%rsp)
    3fba:	4c 2b 74 24 40       	sub    0x40(%rsp),%r14
    3fbf:	8b 44 24 08          	mov    0x8(%rsp),%eax
    3fc3:	48 01 cd             	add    %rcx,%rbp
    3fc6:	48 01 cb             	add    %rcx,%rbx
    3fc9:	3b 44 24 10          	cmp    0x10(%rsp),%eax
    3fcd:	74 61                	je     4030 <jerasure_invert_matrix+0x4e0>
    3fcf:	44 8b 65 00          	mov    0x0(%rbp),%r12d
    3fd3:	45 85 e4             	test   %r12d,%r12d
    3fd6:	74 d8                	je     3fb0 <jerasure_invert_matrix+0x460>
    3fd8:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    3fdd:	c7 45 00 00 00 00 00 	movl   $0x0,0x0(%rbp)
    3fe4:	4c 8d 2c 18          	lea    (%rax,%rbx,1),%r13
    3fe8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    3fef:	00 
    3ff0:	43 8b 74 b5 00       	mov    0x0(%r13,%r14,4),%esi
    3ff5:	44 89 fa             	mov    %r15d,%edx
    3ff8:	44 89 e7             	mov    %r12d,%edi
    3ffb:	e8 10 45 00 00       	callq  8510 <galois_single_multiply>
    4000:	41 31 45 00          	xor    %eax,0x0(%r13)
    4004:	49 83 c5 04          	add    $0x4,%r13
    4008:	4c 39 eb             	cmp    %r13,%rbx
    400b:	75 e3                	jne    3ff0 <jerasure_invert_matrix+0x4a0>
    400d:	eb a1                	jmp    3fb0 <jerasure_invert_matrix+0x460>
    400f:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    4014:	48 81 c4 a8 00 00 00 	add    $0xa8,%rsp
    401b:	5b                   	pop    %rbx
    401c:	5d                   	pop    %rbp
    401d:	41 5c                	pop    %r12
    401f:	41 5d                	pop    %r13
    4021:	41 5e                	pop    %r14
    4023:	41 5f                	pop    %r15
    4025:	c3                   	retq   
    4026:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    402d:	00 00 00 
    4030:	44 8d 70 ff          	lea    -0x1(%rax),%r14d
    4034:	44 89 74 24 10       	mov    %r14d,0x10(%rsp)
    4039:	8b 44 24 10          	mov    0x10(%rsp),%eax
    403d:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
    4042:	48 83 6c 24 38 04    	subq   $0x4,0x38(%rsp)
    4048:	48 01 5c 24 28       	add    %rbx,0x28(%rsp)
    404d:	85 c0                	test   %eax,%eax
    404f:	0f 8f 39 ff ff ff    	jg     3f8e <jerasure_invert_matrix+0x43e>
    4055:	44 8b 74 24 10       	mov    0x10(%rsp),%r14d
    405a:	41 83 ee 01          	sub    $0x1,%r14d
    405e:	79 d4                	jns    4034 <jerasure_invert_matrix+0x4e4>
    4060:	31 c0                	xor    %eax,%eax
    4062:	eb b0                	jmp    4014 <jerasure_invert_matrix+0x4c4>
    4064:	8b 44 24 18          	mov    0x18(%rsp),%eax
    4068:	e9 d9 fd ff ff       	jmpq   3e46 <jerasure_invert_matrix+0x2f6>
    406d:	0f 1f 00             	nopl   (%rax)

0000000000004070 <jerasure_make_decoding_matrix>:
    4070:	f3 0f 1e fa          	endbr64 
    4074:	41 57                	push   %r15
    4076:	41 56                	push   %r14
    4078:	41 55                	push   %r13
    407a:	41 54                	push   %r12
    407c:	41 89 fc             	mov    %edi,%r12d
    407f:	0f af ff             	imul   %edi,%edi
    4082:	55                   	push   %rbp
    4083:	53                   	push   %rbx
    4084:	48 63 ff             	movslq %edi,%rdi
    4087:	48 c1 e7 02          	shl    $0x2,%rdi
    408b:	48 83 ec 28          	sub    $0x28,%rsp
    408f:	89 54 24 14          	mov    %edx,0x14(%rsp)
    4093:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
    4098:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
    409d:	4c 89 4c 24 18       	mov    %r9,0x18(%rsp)
    40a2:	45 85 e4             	test   %r12d,%r12d
    40a5:	0f 8e 02 01 00 00    	jle    41ad <jerasure_make_decoding_matrix+0x13d>
    40ab:	4c 89 c1             	mov    %r8,%rcx
    40ae:	31 c0                	xor    %eax,%eax
    40b0:	31 d2                	xor    %edx,%edx
    40b2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    40b8:	8b 34 81             	mov    (%rcx,%rax,4),%esi
    40bb:	85 f6                	test   %esi,%esi
    40bd:	75 09                	jne    40c8 <jerasure_make_decoding_matrix+0x58>
    40bf:	48 63 f2             	movslq %edx,%rsi
    40c2:	83 c2 01             	add    $0x1,%edx
    40c5:	89 04 b3             	mov    %eax,(%rbx,%rsi,4)
    40c8:	48 83 c0 01          	add    $0x1,%rax
    40cc:	44 39 e2             	cmp    %r12d,%edx
    40cf:	7c e7                	jl     40b8 <jerasure_make_decoding_matrix+0x48>
    40d1:	e8 1a d3 ff ff       	callq  13f0 <malloc@plt>
    40d6:	48 89 c5             	mov    %rax,%rbp
    40d9:	48 85 c0             	test   %rax,%rax
    40dc:	0f 84 d8 00 00 00    	je     41ba <jerasure_make_decoding_matrix+0x14a>
    40e2:	48 89 df             	mov    %rbx,%rdi
    40e5:	41 8d 5c 24 ff       	lea    -0x1(%r12),%ebx
    40ea:	4d 63 f4             	movslq %r12d,%r14
    40ed:	48 89 e9             	mov    %rbp,%rcx
    40f0:	48 8d 04 9d 00 00 00 	lea    0x0(,%rbx,4),%rax
    40f7:	00 
    40f8:	49 c1 e6 02          	shl    $0x2,%r14
    40fc:	45 31 ed             	xor    %r13d,%r13d
    40ff:	48 8d 74 05 04       	lea    0x4(%rbp,%rax,1),%rsi
    4104:	4c 8d 7c 38 04       	lea    0x4(%rax,%rdi,1),%r15
    4109:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    4110:	8b 17                	mov    (%rdi),%edx
    4112:	44 39 e2             	cmp    %r12d,%edx
    4115:	7d 69                	jge    4180 <jerasure_make_decoding_matrix+0x110>
    4117:	48 89 c8             	mov    %rcx,%rax
    411a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    4120:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
    4126:	48 83 c0 04          	add    $0x4,%rax
    412a:	48 39 f0             	cmp    %rsi,%rax
    412d:	75 f1                	jne    4120 <jerasure_make_decoding_matrix+0xb0>
    412f:	44 01 ea             	add    %r13d,%edx
    4132:	48 63 d2             	movslq %edx,%rdx
    4135:	c7 44 95 00 01 00 00 	movl   $0x1,0x0(%rbp,%rdx,4)
    413c:	00 
    413d:	48 83 c7 04          	add    $0x4,%rdi
    4141:	45 01 e5             	add    %r12d,%r13d
    4144:	4c 01 f1             	add    %r14,%rcx
    4147:	4c 01 f6             	add    %r14,%rsi
    414a:	4c 39 ff             	cmp    %r15,%rdi
    414d:	75 c1                	jne    4110 <jerasure_make_decoding_matrix+0xa0>
    414f:	8b 4c 24 14          	mov    0x14(%rsp),%ecx
    4153:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
    4158:	44 89 e2             	mov    %r12d,%edx
    415b:	48 89 ef             	mov    %rbp,%rdi
    415e:	e8 ed f9 ff ff       	callq  3b50 <jerasure_invert_matrix>
    4163:	48 89 ef             	mov    %rbp,%rdi
    4166:	41 89 c4             	mov    %eax,%r12d
    4169:	e8 02 d1 ff ff       	callq  1270 <free@plt>
    416e:	48 83 c4 28          	add    $0x28,%rsp
    4172:	44 89 e0             	mov    %r12d,%eax
    4175:	5b                   	pop    %rbx
    4176:	5d                   	pop    %rbp
    4177:	41 5c                	pop    %r12
    4179:	41 5d                	pop    %r13
    417b:	41 5e                	pop    %r14
    417d:	41 5f                	pop    %r15
    417f:	c3                   	retq   
    4180:	44 29 e2             	sub    %r12d,%edx
    4183:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    4188:	41 0f af d4          	imul   %r12d,%edx
    418c:	48 63 d2             	movslq %edx,%rdx
    418f:	4c 8d 1c 90          	lea    (%rax,%rdx,4),%r11
    4193:	31 c0                	xor    %eax,%eax
    4195:	0f 1f 00             	nopl   (%rax)
    4198:	41 8b 14 83          	mov    (%r11,%rax,4),%edx
    419c:	89 14 81             	mov    %edx,(%rcx,%rax,4)
    419f:	48 89 c2             	mov    %rax,%rdx
    41a2:	48 83 c0 01          	add    $0x1,%rax
    41a6:	48 39 d3             	cmp    %rdx,%rbx
    41a9:	75 ed                	jne    4198 <jerasure_make_decoding_matrix+0x128>
    41ab:	eb 90                	jmp    413d <jerasure_make_decoding_matrix+0xcd>
    41ad:	e8 3e d2 ff ff       	callq  13f0 <malloc@plt>
    41b2:	48 89 c5             	mov    %rax,%rbp
    41b5:	48 85 c0             	test   %rax,%rax
    41b8:	75 95                	jne    414f <jerasure_make_decoding_matrix+0xdf>
    41ba:	41 83 cc ff          	or     $0xffffffff,%r12d
    41be:	eb ae                	jmp    416e <jerasure_make_decoding_matrix+0xfe>

00000000000041c0 <jerasure_invertible_matrix>:
    41c0:	f3 0f 1e fa          	endbr64 
    41c4:	41 57                	push   %r15
    41c6:	41 56                	push   %r14
    41c8:	41 55                	push   %r13
    41ca:	41 54                	push   %r12
    41cc:	55                   	push   %rbp
    41cd:	53                   	push   %rbx
    41ce:	48 83 ec 78          	sub    $0x78,%rsp
    41d2:	48 89 7c 24 28       	mov    %rdi,0x28(%rsp)
    41d7:	89 74 24 14          	mov    %esi,0x14(%rsp)
    41db:	85 f6                	test   %esi,%esi
    41dd:	0f 8e 99 02 00 00    	jle    447c <jerasure_invertible_matrix+0x2bc>
    41e3:	48 63 44 24 14       	movslq 0x14(%rsp),%rax
    41e8:	41 89 d5             	mov    %edx,%r13d
    41eb:	48 c7 44 24 20 00 00 	movq   $0x0,0x20(%rsp)
    41f2:	00 00 
    41f4:	c7 44 24 4c 00 00 00 	movl   $0x0,0x4c(%rsp)
    41fb:	00 
    41fc:	48 8d 1c 85 04 00 00 	lea    0x4(,%rax,4),%rbx
    4203:	00 
    4204:	48 89 c7             	mov    %rax,%rdi
    4207:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    420c:	48 c1 e0 02          	shl    $0x2,%rax
    4210:	48 89 5c 24 68       	mov    %rbx,0x68(%rsp)
    4215:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
    421a:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    421f:	8d 47 ff             	lea    -0x1(%rdi),%eax
    4222:	48 8d 7b 04          	lea    0x4(%rbx),%rdi
    4226:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    422b:	48 8d 04 87          	lea    (%rdi,%rax,4),%rax
    422f:	48 89 5c 24 40       	mov    %rbx,0x40(%rsp)
    4234:	48 89 7c 24 38       	mov    %rdi,0x38(%rsp)
    4239:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    423e:	48 89 5c 24 50       	mov    %rbx,0x50(%rsp)
    4243:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
    424a:	00 00 
    424c:	0f 1f 40 00          	nopl   0x0(%rax)
    4250:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    4255:	8b 30                	mov    (%rax),%esi
    4257:	8b 44 24 30          	mov    0x30(%rsp),%eax
    425b:	8d 58 01             	lea    0x1(%rax),%ebx
    425e:	85 f6                	test   %esi,%esi
    4260:	0f 84 42 01 00 00    	je     43a8 <jerasure_invertible_matrix+0x1e8>
    4266:	83 fe 01             	cmp    $0x1,%esi
    4269:	0f 85 c4 01 00 00    	jne    4433 <jerasure_invertible_matrix+0x273>
    426f:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    4274:	48 39 7c 24 30       	cmp    %rdi,0x30(%rsp)
    4279:	0f 84 fd 01 00 00    	je     447c <jerasure_invertible_matrix+0x2bc>
    427f:	8b 7c 24 14          	mov    0x14(%rsp),%edi
    4283:	01 7c 24 4c          	add    %edi,0x4c(%rsp)
    4287:	8b 44 24 4c          	mov    0x4c(%rsp),%eax
    428b:	48 8b 6c 24 40       	mov    0x40(%rsp),%rbp
    4290:	48 03 6c 24 08       	add    0x8(%rsp),%rbp
    4295:	89 44 24 10          	mov    %eax,0x10(%rsp)
    4299:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    42a0:	44 8b 65 00          	mov    0x0(%rbp),%r12d
    42a4:	45 85 e4             	test   %r12d,%r12d
    42a7:	74 5e                	je     4307 <jerasure_invertible_matrix+0x147>
    42a9:	41 83 fc 01          	cmp    $0x1,%r12d
    42ad:	0f 84 ad 00 00 00    	je     4360 <jerasure_invertible_matrix+0x1a0>
    42b3:	48 63 44 24 10       	movslq 0x10(%rsp),%rax
    42b8:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    42bd:	89 5c 24 48          	mov    %ebx,0x48(%rsp)
    42c1:	4c 8d 34 87          	lea    (%rdi,%rax,4),%r14
    42c5:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    42ca:	48 8d 14 38          	lea    (%rax,%rdi,1),%rdx
    42ce:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
    42d3:	48 8d 14 97          	lea    (%rdi,%rdx,4),%rdx
    42d7:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    42dc:	49 89 d7             	mov    %rdx,%r15
    42df:	48 29 c7             	sub    %rax,%rdi
    42e2:	48 89 fb             	mov    %rdi,%rbx
    42e5:	0f 1f 00             	nopl   (%rax)
    42e8:	41 8b 34 9e          	mov    (%r14,%rbx,4),%esi
    42ec:	44 89 ea             	mov    %r13d,%edx
    42ef:	44 89 e7             	mov    %r12d,%edi
    42f2:	e8 19 42 00 00       	callq  8510 <galois_single_multiply>
    42f7:	41 31 06             	xor    %eax,(%r14)
    42fa:	49 83 c6 04          	add    $0x4,%r14
    42fe:	4d 39 f7             	cmp    %r14,%r15
    4301:	75 e5                	jne    42e8 <jerasure_invertible_matrix+0x128>
    4303:	8b 5c 24 48          	mov    0x48(%rsp),%ebx
    4307:	8b 4c 24 14          	mov    0x14(%rsp),%ecx
    430b:	83 c3 01             	add    $0x1,%ebx
    430e:	01 4c 24 10          	add    %ecx,0x10(%rsp)
    4312:	48 03 6c 24 08       	add    0x8(%rsp),%rbp
    4317:	39 d9                	cmp    %ebx,%ecx
    4319:	75 85                	jne    42a0 <jerasure_invertible_matrix+0xe0>
    431b:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    4320:	48 8b 54 24 68       	mov    0x68(%rsp),%rdx
    4325:	48 01 54 24 40       	add    %rdx,0x40(%rsp)
    432a:	48 8b 54 24 60       	mov    0x60(%rsp),%rdx
    432f:	48 8d 47 01          	lea    0x1(%rdi),%rax
    4333:	48 01 54 24 20       	add    %rdx,0x20(%rsp)
    4338:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
    433d:	48 01 54 24 50       	add    %rdx,0x50(%rsp)
    4342:	48 01 54 24 58       	add    %rdx,0x58(%rsp)
    4347:	48 3b 7c 24 18       	cmp    0x18(%rsp),%rdi
    434c:	0f 84 2a 01 00 00    	je     447c <jerasure_invertible_matrix+0x2bc>
    4352:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    4357:	e9 f4 fe ff ff       	jmpq   4250 <jerasure_invertible_matrix+0x90>
    435c:	0f 1f 40 00          	nopl   0x0(%rax)
    4360:	48 63 54 24 10       	movslq 0x10(%rsp),%rdx
    4365:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    436a:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    436f:	48 8d 0c 3a          	lea    (%rdx,%rdi,1),%rcx
    4373:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
    4378:	48 8d 04 90          	lea    (%rax,%rdx,4),%rax
    437c:	48 8d 34 8f          	lea    (%rdi,%rcx,4),%rsi
    4380:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    4385:	48 29 d1             	sub    %rdx,%rcx
    4388:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    438f:	00 
    4390:	8b 14 88             	mov    (%rax,%rcx,4),%edx
    4393:	31 10                	xor    %edx,(%rax)
    4395:	48 83 c0 04          	add    $0x4,%rax
    4399:	48 39 f0             	cmp    %rsi,%rax
    439c:	75 f2                	jne    4390 <jerasure_invertible_matrix+0x1d0>
    439e:	e9 64 ff ff ff       	jmpq   4307 <jerasure_invertible_matrix+0x147>
    43a3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    43a8:	44 8b 5c 24 14       	mov    0x14(%rsp),%r11d
    43ad:	41 39 db             	cmp    %ebx,%r11d
    43b0:	0f 8e dc 00 00 00    	jle    4492 <jerasure_invertible_matrix+0x2d2>
    43b6:	8b 44 24 4c          	mov    0x4c(%rsp),%eax
    43ba:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    43bf:	48 8b 6c 24 08       	mov    0x8(%rsp),%rbp
    43c4:	44 01 d8             	add    %r11d,%eax
    43c7:	48 63 d0             	movslq %eax,%rdx
    43ca:	48 03 54 24 30       	add    0x30(%rsp),%rdx
    43cf:	48 8d 0c 97          	lea    (%rdi,%rdx,4),%rcx
    43d3:	89 df                	mov    %ebx,%edi
    43d5:	eb 1b                	jmp    43f2 <jerasure_invertible_matrix+0x232>
    43d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    43de:	00 00 
    43e0:	83 c7 01             	add    $0x1,%edi
    43e3:	44 01 d8             	add    %r11d,%eax
    43e6:	48 01 e9             	add    %rbp,%rcx
    43e9:	41 39 fb             	cmp    %edi,%r11d
    43ec:	0f 84 8f 00 00 00    	je     4481 <jerasure_invertible_matrix+0x2c1>
    43f2:	8b 31                	mov    (%rcx),%esi
    43f4:	89 c2                	mov    %eax,%edx
    43f6:	85 f6                	test   %esi,%esi
    43f8:	74 e6                	je     43e0 <jerasure_invertible_matrix+0x220>
    43fa:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    43ff:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    4404:	48 63 d2             	movslq %edx,%rdx
    4407:	48 2b 54 24 20       	sub    0x20(%rsp),%rdx
    440c:	0f 1f 40 00          	nopl   0x0(%rax)
    4410:	8b 08                	mov    (%rax),%ecx
    4412:	8b 34 90             	mov    (%rax,%rdx,4),%esi
    4415:	89 30                	mov    %esi,(%rax)
    4417:	89 0c 90             	mov    %ecx,(%rax,%rdx,4)
    441a:	48 83 c0 04          	add    $0x4,%rax
    441e:	48 39 f8             	cmp    %rdi,%rax
    4421:	75 ed                	jne    4410 <jerasure_invertible_matrix+0x250>
    4423:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    4428:	8b 30                	mov    (%rax),%esi
    442a:	83 fe 01             	cmp    $0x1,%esi
    442d:	0f 84 3c fe ff ff    	je     426f <jerasure_invertible_matrix+0xaf>
    4433:	44 89 ea             	mov    %r13d,%edx
    4436:	bf 01 00 00 00       	mov    $0x1,%edi
    443b:	e8 f0 40 00 00       	callq  8530 <galois_single_divide>
    4440:	4c 8b 64 24 50       	mov    0x50(%rsp),%r12
    4445:	4c 8b 74 24 58       	mov    0x58(%rsp),%r14
    444a:	89 c5                	mov    %eax,%ebp
    444c:	0f 1f 40 00          	nopl   0x0(%rax)
    4450:	41 8b 3c 24          	mov    (%r12),%edi
    4454:	44 89 ea             	mov    %r13d,%edx
    4457:	89 ee                	mov    %ebp,%esi
    4459:	49 83 c4 04          	add    $0x4,%r12
    445d:	e8 ae 40 00 00       	callq  8510 <galois_single_multiply>
    4462:	41 89 44 24 fc       	mov    %eax,-0x4(%r12)
    4467:	4d 39 f4             	cmp    %r14,%r12
    446a:	75 e4                	jne    4450 <jerasure_invertible_matrix+0x290>
    446c:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    4471:	48 39 7c 24 30       	cmp    %rdi,0x30(%rsp)
    4476:	0f 85 03 fe ff ff    	jne    427f <jerasure_invertible_matrix+0xbf>
    447c:	be 01 00 00 00       	mov    $0x1,%esi
    4481:	48 83 c4 78          	add    $0x78,%rsp
    4485:	89 f0                	mov    %esi,%eax
    4487:	5b                   	pop    %rbx
    4488:	5d                   	pop    %rbp
    4489:	41 5c                	pop    %r12
    448b:	41 5d                	pop    %r13
    448d:	41 5e                	pop    %r14
    448f:	41 5f                	pop    %r15
    4491:	c3                   	retq   
    4492:	74 ed                	je     4481 <jerasure_invertible_matrix+0x2c1>
    4494:	8b 54 24 14          	mov    0x14(%rsp),%edx
    4498:	0f af d3             	imul   %ebx,%edx
    449b:	e9 5a ff ff ff       	jmpq   43fa <jerasure_invertible_matrix+0x23a>

00000000000044a0 <jerasure_erasures_to_erased>:
    44a0:	f3 0f 1e fa          	endbr64 
    44a4:	41 54                	push   %r12
    44a6:	41 89 fc             	mov    %edi,%r12d
    44a9:	55                   	push   %rbp
    44aa:	8d 2c 37             	lea    (%rdi,%rsi,1),%ebp
    44ad:	53                   	push   %rbx
    44ae:	48 63 fd             	movslq %ebp,%rdi
    44b1:	48 89 d3             	mov    %rdx,%rbx
    44b4:	48 c1 e7 02          	shl    $0x2,%rdi
    44b8:	e8 33 cf ff ff       	callq  13f0 <malloc@plt>
    44bd:	48 85 c0             	test   %rax,%rax
    44c0:	74 5a                	je     451c <jerasure_erasures_to_erased+0x7c>
    44c2:	85 ed                	test   %ebp,%ebp
    44c4:	7e 21                	jle    44e7 <jerasure_erasures_to_erased+0x47>
    44c6:	8d 4d ff             	lea    -0x1(%rbp),%ecx
    44c9:	48 89 c2             	mov    %rax,%rdx
    44cc:	48 8d 4c 88 04       	lea    0x4(%rax,%rcx,4),%rcx
    44d1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    44d8:	c7 02 00 00 00 00    	movl   $0x0,(%rdx)
    44de:	48 83 c2 04          	add    $0x4,%rdx
    44e2:	48 39 ca             	cmp    %rcx,%rdx
    44e5:	75 f1                	jne    44d8 <jerasure_erasures_to_erased+0x38>
    44e7:	48 63 0b             	movslq (%rbx),%rcx
    44ea:	83 f9 ff             	cmp    $0xffffffff,%ecx
    44ed:	74 2d                	je     451c <jerasure_erasures_to_erased+0x7c>
    44ef:	48 8d 53 04          	lea    0x4(%rbx),%rdx
    44f3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    44f8:	48 8d 0c 88          	lea    (%rax,%rcx,4),%rcx
    44fc:	8b 31                	mov    (%rcx),%esi
    44fe:	85 f6                	test   %esi,%esi
    4500:	75 0e                	jne    4510 <jerasure_erasures_to_erased+0x70>
    4502:	83 ed 01             	sub    $0x1,%ebp
    4505:	c7 01 01 00 00 00    	movl   $0x1,(%rcx)
    450b:	41 39 ec             	cmp    %ebp,%r12d
    450e:	7f 18                	jg     4528 <jerasure_erasures_to_erased+0x88>
    4510:	48 63 0a             	movslq (%rdx),%rcx
    4513:	48 83 c2 04          	add    $0x4,%rdx
    4517:	83 f9 ff             	cmp    $0xffffffff,%ecx
    451a:	75 dc                	jne    44f8 <jerasure_erasures_to_erased+0x58>
    451c:	5b                   	pop    %rbx
    451d:	5d                   	pop    %rbp
    451e:	41 5c                	pop    %r12
    4520:	c3                   	retq   
    4521:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    4528:	48 89 c7             	mov    %rax,%rdi
    452b:	e8 40 cd ff ff       	callq  1270 <free@plt>
    4530:	5b                   	pop    %rbx
    4531:	31 c0                	xor    %eax,%eax
    4533:	5d                   	pop    %rbp
    4534:	41 5c                	pop    %r12
    4536:	c3                   	retq   
    4537:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    453e:	00 00 

0000000000004540 <set_up_ptrs_for_scheduled_decoding>:
    4540:	41 57                	push   %r15
    4542:	41 89 f7             	mov    %esi,%r15d
    4545:	41 56                	push   %r14
    4547:	41 55                	push   %r13
    4549:	4d 89 c5             	mov    %r8,%r13
    454c:	41 54                	push   %r12
    454e:	49 89 cc             	mov    %rcx,%r12
    4551:	55                   	push   %rbp
    4552:	53                   	push   %rbx
    4553:	48 63 df             	movslq %edi,%rbx
    4556:	89 df                	mov    %ebx,%edi
    4558:	48 83 ec 18          	sub    $0x18,%rsp
    455c:	89 74 24 08          	mov    %esi,0x8(%rsp)
    4560:	e8 3b ff ff ff       	callq  44a0 <jerasure_erasures_to_erased>
    4565:	48 85 c0             	test   %rax,%rax
    4568:	0f 84 f2 00 00 00    	je     4660 <set_up_ptrs_for_scheduled_decoding+0x120>
    456e:	48 89 c5             	mov    %rax,%rbp
    4571:	42 8d 04 3b          	lea    (%rbx,%r15,1),%eax
    4575:	48 63 f8             	movslq %eax,%rdi
    4578:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    457c:	48 c1 e7 03          	shl    $0x3,%rdi
    4580:	e8 6b ce ff ff       	callq  13f0 <malloc@plt>
    4585:	49 89 c6             	mov    %rax,%r14
    4588:	85 db                	test   %ebx,%ebx
    458a:	0f 8e d5 00 00 00    	jle    4665 <set_up_ptrs_for_scheduled_decoding+0x125>
    4590:	8d 4b ff             	lea    -0x1(%rbx),%ecx
    4593:	41 89 db             	mov    %ebx,%r11d
    4596:	89 df                	mov    %ebx,%edi
    4598:	31 f6                	xor    %esi,%esi
    459a:	eb 14                	jmp    45b0 <set_up_ptrs_for_scheduled_decoding+0x70>
    459c:	0f 1f 40 00          	nopl   0x0(%rax)
    45a0:	49 89 14 f6          	mov    %rdx,(%r14,%rsi,8)
    45a4:	48 8d 46 01          	lea    0x1(%rsi),%rax
    45a8:	48 39 f1             	cmp    %rsi,%rcx
    45ab:	74 59                	je     4606 <set_up_ptrs_for_scheduled_decoding+0xc6>
    45ad:	48 89 c6             	mov    %rax,%rsi
    45b0:	8b 44 b5 00          	mov    0x0(%rbp,%rsi,4),%eax
    45b4:	49 8b 14 f4          	mov    (%r12,%rsi,8),%rdx
    45b8:	85 c0                	test   %eax,%eax
    45ba:	74 e4                	je     45a0 <set_up_ptrs_for_scheduled_decoding+0x60>
    45bc:	4c 63 ff             	movslq %edi,%r15
    45bf:	8d 47 01             	lea    0x1(%rdi),%eax
    45c2:	46 8b 7c bd 00       	mov    0x0(%rbp,%r15,4),%r15d
    45c7:	48 98                	cltq   
    45c9:	45 85 ff             	test   %r15d,%r15d
    45cc:	74 12                	je     45e0 <set_up_ptrs_for_scheduled_decoding+0xa0>
    45ce:	66 90                	xchg   %ax,%ax
    45d0:	89 c7                	mov    %eax,%edi
    45d2:	48 83 c0 01          	add    $0x1,%rax
    45d6:	44 8b 7c 85 fc       	mov    -0x4(%rbp,%rax,4),%r15d
    45db:	45 85 ff             	test   %r15d,%r15d
    45de:	75 f0                	jne    45d0 <set_up_ptrs_for_scheduled_decoding+0x90>
    45e0:	89 f8                	mov    %edi,%eax
    45e2:	83 c7 01             	add    $0x1,%edi
    45e5:	29 d8                	sub    %ebx,%eax
    45e7:	48 98                	cltq   
    45e9:	49 8b 44 c5 00       	mov    0x0(%r13,%rax,8),%rax
    45ee:	49 89 04 f6          	mov    %rax,(%r14,%rsi,8)
    45f2:	49 63 c3             	movslq %r11d,%rax
    45f5:	41 83 c3 01          	add    $0x1,%r11d
    45f9:	49 89 14 c6          	mov    %rdx,(%r14,%rax,8)
    45fd:	48 8d 46 01          	lea    0x1(%rsi),%rax
    4601:	48 39 f1             	cmp    %rsi,%rcx
    4604:	75 a7                	jne    45ad <set_up_ptrs_for_scheduled_decoding+0x6d>
    4606:	39 5c 24 0c          	cmp    %ebx,0xc(%rsp)
    460a:	7e 3a                	jle    4646 <set_up_ptrs_for_scheduled_decoding+0x106>
    460c:	8b 44 24 08          	mov    0x8(%rsp),%eax
    4610:	48 8d 4c 9d 00       	lea    0x0(%rbp,%rbx,4),%rcx
    4615:	8d 70 ff             	lea    -0x1(%rax),%esi
    4618:	31 c0                	xor    %eax,%eax
    461a:	eb 07                	jmp    4623 <set_up_ptrs_for_scheduled_decoding+0xe3>
    461c:	0f 1f 40 00          	nopl   0x0(%rax)
    4620:	48 89 d0             	mov    %rdx,%rax
    4623:	8b 14 81             	mov    (%rcx,%rax,4),%edx
    4626:	85 d2                	test   %edx,%edx
    4628:	74 13                	je     463d <set_up_ptrs_for_scheduled_decoding+0xfd>
    462a:	48 63 f8             	movslq %eax,%rdi
    462d:	49 63 d3             	movslq %r11d,%rdx
    4630:	41 83 c3 01          	add    $0x1,%r11d
    4634:	49 8b 7c fd 00       	mov    0x0(%r13,%rdi,8),%rdi
    4639:	49 89 3c d6          	mov    %rdi,(%r14,%rdx,8)
    463d:	48 8d 50 01          	lea    0x1(%rax),%rdx
    4641:	48 39 c6             	cmp    %rax,%rsi
    4644:	75 da                	jne    4620 <set_up_ptrs_for_scheduled_decoding+0xe0>
    4646:	48 89 ef             	mov    %rbp,%rdi
    4649:	e8 22 cc ff ff       	callq  1270 <free@plt>
    464e:	48 83 c4 18          	add    $0x18,%rsp
    4652:	4c 89 f0             	mov    %r14,%rax
    4655:	5b                   	pop    %rbx
    4656:	5d                   	pop    %rbp
    4657:	41 5c                	pop    %r12
    4659:	41 5d                	pop    %r13
    465b:	41 5e                	pop    %r14
    465d:	41 5f                	pop    %r15
    465f:	c3                   	retq   
    4660:	45 31 f6             	xor    %r14d,%r14d
    4663:	eb e9                	jmp    464e <set_up_ptrs_for_scheduled_decoding+0x10e>
    4665:	41 89 db             	mov    %ebx,%r11d
    4668:	eb 9c                	jmp    4606 <set_up_ptrs_for_scheduled_decoding+0xc6>
    466a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000004670 <jerasure_free_schedule>:
    4670:	f3 0f 1e fa          	endbr64 
    4674:	55                   	push   %rbp
    4675:	48 89 fd             	mov    %rdi,%rbp
    4678:	53                   	push   %rbx
    4679:	48 83 ec 08          	sub    $0x8,%rsp
    467d:	48 8b 3f             	mov    (%rdi),%rdi
    4680:	8b 17                	mov    (%rdi),%edx
    4682:	85 d2                	test   %edx,%edx
    4684:	78 1c                	js     46a2 <jerasure_free_schedule+0x32>
    4686:	48 8d 5d 08          	lea    0x8(%rbp),%rbx
    468a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    4690:	e8 db cb ff ff       	callq  1270 <free@plt>
    4695:	48 8b 3b             	mov    (%rbx),%rdi
    4698:	48 83 c3 08          	add    $0x8,%rbx
    469c:	8b 07                	mov    (%rdi),%eax
    469e:	85 c0                	test   %eax,%eax
    46a0:	79 ee                	jns    4690 <jerasure_free_schedule+0x20>
    46a2:	e8 c9 cb ff ff       	callq  1270 <free@plt>
    46a7:	48 83 c4 08          	add    $0x8,%rsp
    46ab:	48 89 ef             	mov    %rbp,%rdi
    46ae:	5b                   	pop    %rbx
    46af:	5d                   	pop    %rbp
    46b0:	e9 bb cb ff ff       	jmpq   1270 <free@plt>
    46b5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    46bc:	00 00 00 00 

00000000000046c0 <jerasure_free_schedule_cache>:
    46c0:	f3 0f 1e fa          	endbr64 
    46c4:	41 57                	push   %r15
    46c6:	41 56                	push   %r14
    46c8:	41 55                	push   %r13
    46ca:	41 54                	push   %r12
    46cc:	55                   	push   %rbp
    46cd:	53                   	push   %rbx
    46ce:	48 83 ec 28          	sub    $0x28,%rsp
    46d2:	48 89 54 24 18       	mov    %rdx,0x18(%rsp)
    46d7:	83 fe 02             	cmp    $0x2,%esi
    46da:	0f 85 88 00 00 00    	jne    4768 <jerasure_free_schedule_cache+0xa8>
    46e0:	8d 47 02             	lea    0x2(%rdi),%eax
    46e3:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    46e7:	85 c0                	test   %eax,%eax
    46e9:	7e 65                	jle    4750 <jerasure_free_schedule_cache+0x90>
    46eb:	4c 63 f0             	movslq %eax,%r14
    46ee:	31 ed                	xor    %ebp,%ebp
    46f0:	4e 8d 3c f5 00 00 00 	lea    0x0(,%r14,8),%r15
    46f7:	00 
    46f8:	49 f7 de             	neg    %r14
    46fb:	49 8d 47 08          	lea    0x8(%r15),%rax
    46ff:	4e 8d 24 3a          	lea    (%rdx,%r15,1),%r12
    4703:	49 c1 e6 03          	shl    $0x3,%r14
    4707:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    470c:	4a 8d 5c 3a 08       	lea    0x8(%rdx,%r15,1),%rbx
    4711:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    4718:	4a 8b 7c 33 f8       	mov    -0x8(%rbx,%r14,1),%rdi
    471d:	83 c5 01             	add    $0x1,%ebp
    4720:	e8 4b ff ff ff       	callq  4670 <jerasure_free_schedule>
    4725:	3b 6c 24 0c          	cmp    0xc(%rsp),%ebp
    4729:	74 25                	je     4750 <jerasure_free_schedule_cache+0x90>
    472b:	4d 89 e5             	mov    %r12,%r13
    472e:	66 90                	xchg   %ax,%ax
    4730:	49 8b 7d 00          	mov    0x0(%r13),%rdi
    4734:	49 83 c5 08          	add    $0x8,%r13
    4738:	e8 33 ff ff ff       	callq  4670 <jerasure_free_schedule>
    473d:	49 39 dd             	cmp    %rbx,%r13
    4740:	75 ee                	jne    4730 <jerasure_free_schedule_cache+0x70>
    4742:	4d 01 fc             	add    %r15,%r12
    4745:	48 03 5c 24 10       	add    0x10(%rsp),%rbx
    474a:	eb cc                	jmp    4718 <jerasure_free_schedule_cache+0x58>
    474c:	0f 1f 40 00          	nopl   0x0(%rax)
    4750:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    4755:	48 83 c4 28          	add    $0x28,%rsp
    4759:	5b                   	pop    %rbx
    475a:	5d                   	pop    %rbp
    475b:	41 5c                	pop    %r12
    475d:	41 5d                	pop    %r13
    475f:	41 5e                	pop    %r14
    4761:	41 5f                	pop    %r15
    4763:	e9 08 cb ff ff       	jmpq   1270 <free@plt>
    4768:	48 8b 0d d1 c9 00 00 	mov    0xc9d1(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    476f:	ba 2f 00 00 00       	mov    $0x2f,%edx
    4774:	be 01 00 00 00       	mov    $0x1,%esi
    4779:	48 8d 3d b0 5f 00 00 	lea    0x5fb0(%rip),%rdi        # a730 <__PRETTY_FUNCTION__.5230+0x5f>
    4780:	e8 db cc ff ff       	callq  1460 <fwrite@plt>
    4785:	bf 01 00 00 00       	mov    $0x1,%edi
    478a:	e8 c1 cc ff ff       	callq  1450 <exit@plt>
    478f:	90                   	nop

0000000000004790 <jerasure_matrix_dotprod>:
    4790:	f3 0f 1e fa          	endbr64 
    4794:	41 57                	push   %r15
    4796:	41 56                	push   %r14
    4798:	41 55                	push   %r13
    479a:	41 54                	push   %r12
    479c:	55                   	push   %rbp
    479d:	53                   	push   %rbx
    479e:	48 83 ec 28          	sub    $0x28,%rsp
    47a2:	89 7c 24 18          	mov    %edi,0x18(%rsp)
    47a6:	44 8b 7c 24 68       	mov    0x68(%rsp),%r15d
    47ab:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
    47b0:	83 fe 20             	cmp    $0x20,%esi
    47b3:	0f 87 af 01 00 00    	ja     4968 <jerasure_matrix_dotprod+0x1d8>
    47b9:	48 89 d5             	mov    %rdx,%rbp
    47bc:	41 89 f4             	mov    %esi,%r12d
    47bf:	48 ba 02 01 01 00 01 	movabs $0x100010102,%rdx
    47c6:	00 00 00 
    47c9:	48 0f a3 f2          	bt     %rsi,%rdx
    47cd:	0f 83 95 01 00 00    	jae    4968 <jerasure_matrix_dotprod+0x1d8>
    47d3:	49 89 ce             	mov    %rcx,%r14
    47d6:	49 63 c0             	movslq %r8d,%rax
    47d9:	3b 44 24 18          	cmp    0x18(%rsp),%eax
    47dd:	0f 8c ad 01 00 00    	jl     4990 <jerasure_matrix_dotprod+0x200>
    47e3:	48 8b 4c 24 60       	mov    0x60(%rsp),%rcx
    47e8:	2b 44 24 18          	sub    0x18(%rsp),%eax
    47ec:	48 98                	cltq   
    47ee:	4c 8b 1c c1          	mov    (%rcx,%rax,8),%r11
    47f2:	8b 44 24 18          	mov    0x18(%rsp),%eax
    47f6:	85 c0                	test   %eax,%eax
    47f8:	0f 8e a7 01 00 00    	jle    49a5 <jerasure_matrix_dotprod+0x215>
    47fe:	49 63 cf             	movslq %r15d,%rcx
    4801:	44 8d 68 ff          	lea    -0x1(%rax),%r13d
    4805:	31 c0                	xor    %eax,%eax
    4807:	44 89 64 24 1c       	mov    %r12d,0x1c(%rsp)
    480c:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    4811:	31 db                	xor    %ebx,%ebx
    4813:	44 89 f9             	mov    %r15d,%ecx
    4816:	41 89 c4             	mov    %eax,%r12d
    4819:	4d 89 df             	mov    %r11,%r15
    481c:	eb 0e                	jmp    482c <jerasure_matrix_dotprod+0x9c>
    481e:	66 90                	xchg   %ax,%ax
    4820:	48 8d 53 01          	lea    0x1(%rbx),%rdx
    4824:	49 39 dd             	cmp    %rbx,%r13
    4827:	74 77                	je     48a0 <jerasure_matrix_dotprod+0x110>
    4829:	48 89 d3             	mov    %rdx,%rbx
    482c:	83 7c 9d 00 01       	cmpl   $0x1,0x0(%rbp,%rbx,4)
    4831:	75 ed                	jne    4820 <jerasure_matrix_dotprod+0x90>
    4833:	4d 85 f6             	test   %r14,%r14
    4836:	0f 84 04 02 00 00    	je     4a40 <jerasure_matrix_dotprod+0x2b0>
    483c:	49 63 14 9e          	movslq (%r14,%rbx,4),%rdx
    4840:	3b 54 24 18          	cmp    0x18(%rsp),%edx
    4844:	0f 8d 9e 01 00 00    	jge    49e8 <jerasure_matrix_dotprod+0x258>
    484a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    484f:	48 8b 3c d0          	mov    (%rax,%rdx,8),%rdi
    4853:	89 4c 24 68          	mov    %ecx,0x68(%rsp)
    4857:	45 85 e4             	test   %r12d,%r12d
    485a:	0f 85 58 01 00 00    	jne    49b8 <jerasure_matrix_dotprod+0x228>
    4860:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    4865:	48 89 fe             	mov    %rdi,%rsi
    4868:	4c 89 ff             	mov    %r15,%rdi
    486b:	41 bc 01 00 00 00    	mov    $0x1,%r12d
    4871:	e8 3a cb ff ff       	callq  13b0 <memcpy@plt>
    4876:	8b 4c 24 68          	mov    0x68(%rsp),%ecx
    487a:	66 0f ef c0          	pxor   %xmm0,%xmm0
    487e:	48 8d 53 01          	lea    0x1(%rbx),%rdx
    4882:	f2 0f 2a c1          	cvtsi2sd %ecx,%xmm0
    4886:	f2 0f 58 05 c2 c8 00 	addsd  0xc8c2(%rip),%xmm0        # 11150 <jerasure_total_memcpy_bytes>
    488d:	00 
    488e:	f2 0f 11 05 ba c8 00 	movsd  %xmm0,0xc8ba(%rip)        # 11150 <jerasure_total_memcpy_bytes>
    4895:	00 
    4896:	49 39 dd             	cmp    %rbx,%r13
    4899:	75 8e                	jne    4829 <jerasure_matrix_dotprod+0x99>
    489b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    48a0:	44 89 e0             	mov    %r12d,%eax
    48a3:	4d 89 fb             	mov    %r15,%r11
    48a6:	41 89 cf             	mov    %ecx,%r15d
    48a9:	44 8b 64 24 1c       	mov    0x1c(%rsp),%r12d
    48ae:	44 89 7c 24 68       	mov    %r15d,0x68(%rsp)
    48b3:	31 db                	xor    %ebx,%ebx
    48b5:	4d 89 ef             	mov    %r13,%r15
    48b8:	89 c1                	mov    %eax,%ecx
    48ba:	4c 89 5c 24 10       	mov    %r11,0x10(%rsp)
    48bf:	44 8b 6c 24 68       	mov    0x68(%rsp),%r13d
    48c4:	eb 5b                	jmp    4921 <jerasure_matrix_dotprod+0x191>
    48c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    48cd:	00 00 00 
    48d0:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    48d5:	48 8b 3c f8          	mov    (%rax,%rdi,8),%rdi
    48d9:	41 83 fc 10          	cmp    $0x10,%r12d
    48dd:	74 74                	je     4953 <jerasure_matrix_dotprod+0x1c3>
    48df:	41 83 fc 20          	cmp    $0x20,%r12d
    48e3:	0f 84 3f 01 00 00    	je     4a28 <jerasure_matrix_dotprod+0x298>
    48e9:	41 83 fc 08          	cmp    $0x8,%r12d
    48ed:	0f 84 1d 01 00 00    	je     4a10 <jerasure_matrix_dotprod+0x280>
    48f3:	66 0f ef c0          	pxor   %xmm0,%xmm0
    48f7:	b9 01 00 00 00       	mov    $0x1,%ecx
    48fc:	f2 41 0f 2a c5       	cvtsi2sd %r13d,%xmm0
    4901:	f2 0f 58 05 4f c8 00 	addsd  0xc84f(%rip),%xmm0        # 11158 <jerasure_total_gf_bytes>
    4908:	00 
    4909:	f2 0f 11 05 47 c8 00 	movsd  %xmm0,0xc847(%rip)        # 11158 <jerasure_total_gf_bytes>
    4910:	00 
    4911:	48 8d 73 01          	lea    0x1(%rbx),%rsi
    4915:	49 39 df             	cmp    %rbx,%r15
    4918:	0f 84 87 00 00 00    	je     49a5 <jerasure_matrix_dotprod+0x215>
    491e:	48 89 f3             	mov    %rsi,%rbx
    4921:	8b 74 9d 00          	mov    0x0(%rbp,%rbx,4),%esi
    4925:	83 fe 01             	cmp    $0x1,%esi
    4928:	76 e7                	jbe    4911 <jerasure_matrix_dotprod+0x181>
    492a:	4d 85 f6             	test   %r14,%r14
    492d:	0f 84 cd 00 00 00    	je     4a00 <jerasure_matrix_dotprod+0x270>
    4933:	49 63 3c 9e          	movslq (%r14,%rbx,4),%rdi
    4937:	3b 7c 24 18          	cmp    0x18(%rsp),%edi
    493b:	7c 93                	jl     48d0 <jerasure_matrix_dotprod+0x140>
    493d:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    4942:	2b 7c 24 18          	sub    0x18(%rsp),%edi
    4946:	48 63 ff             	movslq %edi,%rdi
    4949:	48 8b 3c f8          	mov    (%rax,%rdi,8),%rdi
    494d:	41 83 fc 10          	cmp    $0x10,%r12d
    4951:	75 8c                	jne    48df <jerasure_matrix_dotprod+0x14f>
    4953:	41 89 c8             	mov    %ecx,%r8d
    4956:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    495b:	44 89 ea             	mov    %r13d,%edx
    495e:	e8 cd 30 00 00       	callq  7a30 <galois_w16_region_multiply>
    4963:	eb 8e                	jmp    48f3 <jerasure_matrix_dotprod+0x163>
    4965:	0f 1f 00             	nopl   (%rax)
    4968:	48 8b 0d d1 c7 00 00 	mov    0xc7d1(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    496f:	ba 44 00 00 00       	mov    $0x44,%edx
    4974:	be 01 00 00 00       	mov    $0x1,%esi
    4979:	48 8d 3d e0 5d 00 00 	lea    0x5de0(%rip),%rdi        # a760 <__PRETTY_FUNCTION__.5230+0x8f>
    4980:	e8 db ca ff ff       	callq  1460 <fwrite@plt>
    4985:	bf 01 00 00 00       	mov    $0x1,%edi
    498a:	e8 c1 ca ff ff       	callq  1450 <exit@plt>
    498f:	90                   	nop
    4990:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    4995:	4c 8b 1c c1          	mov    (%rcx,%rax,8),%r11
    4999:	8b 44 24 18          	mov    0x18(%rsp),%eax
    499d:	85 c0                	test   %eax,%eax
    499f:	0f 8f 59 fe ff ff    	jg     47fe <jerasure_matrix_dotprod+0x6e>
    49a5:	48 83 c4 28          	add    $0x28,%rsp
    49a9:	5b                   	pop    %rbx
    49aa:	5d                   	pop    %rbp
    49ab:	41 5c                	pop    %r12
    49ad:	41 5d                	pop    %r13
    49af:	41 5e                	pop    %r14
    49b1:	41 5f                	pop    %r15
    49b3:	c3                   	retq   
    49b4:	0f 1f 40 00          	nopl   0x0(%rax)
    49b8:	4c 89 fa             	mov    %r15,%rdx
    49bb:	4c 89 fe             	mov    %r15,%rsi
    49be:	e8 1d 36 00 00       	callq  7fe0 <galois_region_xor>
    49c3:	8b 4c 24 68          	mov    0x68(%rsp),%ecx
    49c7:	66 0f ef c0          	pxor   %xmm0,%xmm0
    49cb:	f2 0f 2a c1          	cvtsi2sd %ecx,%xmm0
    49cf:	f2 0f 58 05 89 c7 00 	addsd  0xc789(%rip),%xmm0        # 11160 <jerasure_total_xor_bytes>
    49d6:	00 
    49d7:	f2 0f 11 05 81 c7 00 	movsd  %xmm0,0xc781(%rip)        # 11160 <jerasure_total_xor_bytes>
    49de:	00 
    49df:	e9 3c fe ff ff       	jmpq   4820 <jerasure_matrix_dotprod+0x90>
    49e4:	0f 1f 40 00          	nopl   0x0(%rax)
    49e8:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    49ed:	2b 54 24 18          	sub    0x18(%rsp),%edx
    49f1:	48 63 d2             	movslq %edx,%rdx
    49f4:	48 8b 3c d0          	mov    (%rax,%rdx,8),%rdi
    49f8:	e9 56 fe ff ff       	jmpq   4853 <jerasure_matrix_dotprod+0xc3>
    49fd:	0f 1f 00             	nopl   (%rax)
    4a00:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    4a05:	48 8b 3c d8          	mov    (%rax,%rbx,8),%rdi
    4a09:	e9 cb fe ff ff       	jmpq   48d9 <jerasure_matrix_dotprod+0x149>
    4a0e:	66 90                	xchg   %ax,%ax
    4a10:	41 89 c8             	mov    %ecx,%r8d
    4a13:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    4a18:	44 89 ea             	mov    %r13d,%edx
    4a1b:	e8 a0 2e 00 00       	callq  78c0 <galois_w08_region_multiply>
    4a20:	e9 ce fe ff ff       	jmpq   48f3 <jerasure_matrix_dotprod+0x163>
    4a25:	0f 1f 00             	nopl   (%rax)
    4a28:	41 89 c8             	mov    %ecx,%r8d
    4a2b:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    4a30:	44 89 ea             	mov    %r13d,%edx
    4a33:	e8 38 37 00 00       	callq  8170 <galois_w32_region_multiply>
    4a38:	e9 b6 fe ff ff       	jmpq   48f3 <jerasure_matrix_dotprod+0x163>
    4a3d:	0f 1f 00             	nopl   (%rax)
    4a40:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    4a45:	48 8b 3c d8          	mov    (%rax,%rbx,8),%rdi
    4a49:	e9 05 fe ff ff       	jmpq   4853 <jerasure_matrix_dotprod+0xc3>
    4a4e:	66 90                	xchg   %ax,%ax

0000000000004a50 <jerasure_matrix_decode>:
    4a50:	f3 0f 1e fa          	endbr64 
    4a54:	41 57                	push   %r15
    4a56:	41 89 d7             	mov    %edx,%r15d
    4a59:	4c 89 ca             	mov    %r9,%rdx
    4a5c:	41 56                	push   %r14
    4a5e:	41 8d 47 f8          	lea    -0x8(%r15),%eax
    4a62:	41 55                	push   %r13
    4a64:	41 54                	push   %r12
    4a66:	55                   	push   %rbp
    4a67:	44 89 c5             	mov    %r8d,%ebp
    4a6a:	53                   	push   %rbx
    4a6b:	89 fb                	mov    %edi,%ebx
    4a6d:	48 83 ec 48          	sub    $0x48,%rsp
    4a71:	83 e0 f7             	and    $0xfffffff7,%eax
    4a74:	89 74 24 38          	mov    %esi,0x38(%rsp)
    4a78:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    4a7d:	74 0a                	je     4a89 <jerasure_matrix_decode+0x39>
    4a7f:	41 83 ff 20          	cmp    $0x20,%r15d
    4a83:	0f 85 29 04 00 00    	jne    4eb2 <jerasure_matrix_decode+0x462>
    4a89:	8b 74 24 38          	mov    0x38(%rsp),%esi
    4a8d:	89 df                	mov    %ebx,%edi
    4a8f:	e8 0c fa ff ff       	callq  44a0 <jerasure_erasures_to_erased>
    4a94:	49 89 c6             	mov    %rax,%r14
    4a97:	48 85 c0             	test   %rax,%rax
    4a9a:	0f 84 12 04 00 00    	je     4eb2 <jerasure_matrix_decode+0x462>
    4aa0:	85 db                	test   %ebx,%ebx
    4aa2:	0f 8e a8 03 00 00    	jle    4e50 <jerasure_matrix_decode+0x400>
    4aa8:	8d 4b ff             	lea    -0x1(%rbx),%ecx
    4aab:	41 89 db             	mov    %ebx,%r11d
    4aae:	31 c0                	xor    %eax,%eax
    4ab0:	45 31 ed             	xor    %r13d,%r13d
    4ab3:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
    4ab7:	eb 0a                	jmp    4ac3 <jerasure_matrix_decode+0x73>
    4ab9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    4ac0:	48 89 d0             	mov    %rdx,%rax
    4ac3:	41 8b 14 86          	mov    (%r14,%rax,4),%edx
    4ac7:	85 d2                	test   %edx,%edx
    4ac9:	74 07                	je     4ad2 <jerasure_matrix_decode+0x82>
    4acb:	41 83 c5 01          	add    $0x1,%r13d
    4acf:	41 89 c3             	mov    %eax,%r11d
    4ad2:	48 8d 50 01          	lea    0x1(%rax),%rdx
    4ad6:	48 39 c1             	cmp    %rax,%rcx
    4ad9:	75 e5                	jne    4ac0 <jerasure_matrix_decode+0x70>
    4adb:	85 ed                	test   %ebp,%ebp
    4add:	0f 84 4d 02 00 00    	je     4d30 <jerasure_matrix_decode+0x2e0>
    4ae3:	48 63 c3             	movslq %ebx,%rax
    4ae6:	41 8b 04 86          	mov    (%r14,%rax,4),%eax
    4aea:	85 c0                	test   %eax,%eax
    4aec:	44 0f 45 db          	cmovne %ebx,%r11d
    4af0:	41 83 fd 01          	cmp    $0x1,%r13d
    4af4:	0f 8f 43 02 00 00    	jg     4d3d <jerasure_matrix_decode+0x2ed>
    4afa:	0f 85 30 03 00 00    	jne    4e30 <jerasure_matrix_decode+0x3e0>
    4b00:	48 63 c3             	movslq %ebx,%rax
    4b03:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    4b08:	48 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%rdi
    4b0f:	00 
    4b10:	85 ed                	test   %ebp,%ebp
    4b12:	0f 85 c0 02 00 00    	jne    4dd8 <jerasure_matrix_decode+0x388>
    4b18:	44 89 5c 24 10       	mov    %r11d,0x10(%rsp)
    4b1d:	e8 ce c8 ff ff       	callq  13f0 <malloc@plt>
    4b22:	44 8b 5c 24 10       	mov    0x10(%rsp),%r11d
    4b27:	48 85 c0             	test   %rax,%rax
    4b2a:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    4b2f:	0f 84 75 03 00 00    	je     4eaa <jerasure_matrix_decode+0x45a>
    4b35:	89 df                	mov    %ebx,%edi
    4b37:	44 89 5c 24 18       	mov    %r11d,0x18(%rsp)
    4b3c:	0f af fb             	imul   %ebx,%edi
    4b3f:	48 63 ff             	movslq %edi,%rdi
    4b42:	48 c1 e7 02          	shl    $0x2,%rdi
    4b46:	e8 a5 c8 ff ff       	callq  13f0 <malloc@plt>
    4b4b:	44 8b 5c 24 18       	mov    0x18(%rsp),%r11d
    4b50:	48 85 c0             	test   %rax,%rax
    4b53:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    4b58:	0f 84 86 03 00 00    	je     4ee4 <jerasure_matrix_decode+0x494>
    4b5e:	44 89 5c 24 18       	mov    %r11d,0x18(%rsp)
    4b63:	48 83 ec 08          	sub    $0x8,%rsp
    4b67:	4d 89 f0             	mov    %r14,%r8
    4b6a:	44 89 fa             	mov    %r15d,%edx
    4b6d:	ff 74 24 10          	pushq  0x10(%rsp)
    4b71:	4c 8b 4c 24 20       	mov    0x20(%rsp),%r9
    4b76:	89 df                	mov    %ebx,%edi
    4b78:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    4b7d:	8b 74 24 48          	mov    0x48(%rsp),%esi
    4b81:	e8 ea f4 ff ff       	callq  4070 <jerasure_make_decoding_matrix>
    4b86:	41 5b                	pop    %r11
    4b88:	5d                   	pop    %rbp
    4b89:	85 c0                	test   %eax,%eax
    4b8b:	0f 88 2c 03 00 00    	js     4ebd <jerasure_matrix_decode+0x46d>
    4b91:	44 8b 5c 24 18       	mov    0x18(%rsp),%r11d
    4b96:	45 85 db             	test   %r11d,%r11d
    4b99:	0f 8e b9 01 00 00    	jle    4d58 <jerasure_matrix_decode+0x308>
    4b9f:	31 c0                	xor    %eax,%eax
    4ba1:	4c 89 74 24 30       	mov    %r14,0x30(%rsp)
    4ba6:	4d 89 f4             	mov    %r14,%r12
    4ba9:	44 89 fe             	mov    %r15d,%esi
    4bac:	41 89 de             	mov    %ebx,%r14d
    4baf:	45 89 ef             	mov    %r13d,%r15d
    4bb2:	31 ed                	xor    %ebp,%ebp
    4bb4:	89 c3                	mov    %eax,%ebx
    4bb6:	45 89 dd             	mov    %r11d,%r13d
    4bb9:	eb 19                	jmp    4bd4 <jerasure_matrix_decode+0x184>
    4bbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    4bc0:	83 c5 01             	add    $0x1,%ebp
    4bc3:	49 83 c4 04          	add    $0x4,%r12
    4bc7:	44 01 f3             	add    %r14d,%ebx
    4bca:	45 85 ff             	test   %r15d,%r15d
    4bcd:	7e 61                	jle    4c30 <jerasure_matrix_decode+0x1e0>
    4bcf:	41 39 ed             	cmp    %ebp,%r13d
    4bd2:	7e 5c                	jle    4c30 <jerasure_matrix_decode+0x1e0>
    4bd4:	41 8b 0c 24          	mov    (%r12),%ecx
    4bd8:	85 c9                	test   %ecx,%ecx
    4bda:	74 e4                	je     4bc0 <jerasure_matrix_decode+0x170>
    4bdc:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    4be1:	48 63 d3             	movslq %ebx,%rdx
    4be4:	41 89 e8             	mov    %ebp,%r8d
    4be7:	44 89 f7             	mov    %r14d,%edi
    4bea:	41 83 ef 01          	sub    $0x1,%r15d
    4bee:	83 c5 01             	add    $0x1,%ebp
    4bf1:	49 83 c4 04          	add    $0x4,%r12
    4bf5:	44 01 f3             	add    %r14d,%ebx
    4bf8:	48 8d 14 90          	lea    (%rax,%rdx,4),%rdx
    4bfc:	8b 84 24 90 00 00 00 	mov    0x90(%rsp),%eax
    4c03:	50                   	push   %rax
    4c04:	ff b4 24 90 00 00 00 	pushq  0x90(%rsp)
    4c0b:	4c 8b 8c 24 90 00 00 	mov    0x90(%rsp),%r9
    4c12:	00 
    4c13:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
    4c18:	89 74 24 28          	mov    %esi,0x28(%rsp)
    4c1c:	e8 6f fb ff ff       	callq  4790 <jerasure_matrix_dotprod>
    4c21:	58                   	pop    %rax
    4c22:	5a                   	pop    %rdx
    4c23:	8b 74 24 18          	mov    0x18(%rsp),%esi
    4c27:	45 85 ff             	test   %r15d,%r15d
    4c2a:	7f a3                	jg     4bcf <jerasure_matrix_decode+0x17f>
    4c2c:	0f 1f 40 00          	nopl   0x0(%rax)
    4c30:	45 89 eb             	mov    %r13d,%r11d
    4c33:	45 89 fd             	mov    %r15d,%r13d
    4c36:	44 89 f3             	mov    %r14d,%ebx
    4c39:	41 89 f7             	mov    %esi,%r15d
    4c3c:	4c 8b 74 24 30       	mov    0x30(%rsp),%r14
    4c41:	45 85 ed             	test   %r13d,%r13d
    4c44:	0f 85 0e 01 00 00    	jne    4d58 <jerasure_matrix_decode+0x308>
    4c4a:	8b 7c 24 38          	mov    0x38(%rsp),%edi
    4c4e:	48 63 c3             	movslq %ebx,%rax
    4c51:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    4c56:	85 ff                	test   %edi,%edi
    4c58:	0f 8e 91 00 00 00    	jle    4cef <jerasure_matrix_decode+0x29f>
    4c5e:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    4c63:	45 31 ed             	xor    %r13d,%r13d
    4c66:	4c 89 74 24 18       	mov    %r14,0x18(%rsp)
    4c6b:	89 dd                	mov    %ebx,%ebp
    4c6d:	4d 8d 24 86          	lea    (%r14,%rax,4),%r12
    4c71:	8b 44 24 38          	mov    0x38(%rsp),%eax
    4c75:	45 89 ee             	mov    %r13d,%r14d
    4c78:	01 d8                	add    %ebx,%eax
    4c7a:	41 89 c5             	mov    %eax,%r13d
    4c7d:	44 89 f8             	mov    %r15d,%eax
    4c80:	41 89 df             	mov    %ebx,%r15d
    4c83:	89 c3                	mov    %eax,%ebx
    4c85:	eb 18                	jmp    4c9f <jerasure_matrix_decode+0x24f>
    4c87:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    4c8e:	00 00 
    4c90:	83 c5 01             	add    $0x1,%ebp
    4c93:	49 83 c4 04          	add    $0x4,%r12
    4c97:	45 01 fe             	add    %r15d,%r14d
    4c9a:	41 39 ed             	cmp    %ebp,%r13d
    4c9d:	74 4b                	je     4cea <jerasure_matrix_decode+0x29a>
    4c9f:	41 8b 34 24          	mov    (%r12),%esi
    4ca3:	85 f6                	test   %esi,%esi
    4ca5:	74 e9                	je     4c90 <jerasure_matrix_decode+0x240>
    4ca7:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    4cac:	49 63 d6             	movslq %r14d,%rdx
    4caf:	41 89 e8             	mov    %ebp,%r8d
    4cb2:	31 c9                	xor    %ecx,%ecx
    4cb4:	89 de                	mov    %ebx,%esi
    4cb6:	44 89 ff             	mov    %r15d,%edi
    4cb9:	83 c5 01             	add    $0x1,%ebp
    4cbc:	49 83 c4 04          	add    $0x4,%r12
    4cc0:	48 8d 14 90          	lea    (%rax,%rdx,4),%rdx
    4cc4:	8b 84 24 90 00 00 00 	mov    0x90(%rsp),%eax
    4ccb:	45 01 fe             	add    %r15d,%r14d
    4cce:	50                   	push   %rax
    4ccf:	ff b4 24 90 00 00 00 	pushq  0x90(%rsp)
    4cd6:	4c 8b 8c 24 90 00 00 	mov    0x90(%rsp),%r9
    4cdd:	00 
    4cde:	e8 ad fa ff ff       	callq  4790 <jerasure_matrix_dotprod>
    4ce3:	5a                   	pop    %rdx
    4ce4:	59                   	pop    %rcx
    4ce5:	41 39 ed             	cmp    %ebp,%r13d
    4ce8:	75 b5                	jne    4c9f <jerasure_matrix_decode+0x24f>
    4cea:	4c 8b 74 24 18       	mov    0x18(%rsp),%r14
    4cef:	4c 89 f7             	mov    %r14,%rdi
    4cf2:	e8 79 c5 ff ff       	callq  1270 <free@plt>
    4cf7:	48 83 7c 24 08 00    	cmpq   $0x0,0x8(%rsp)
    4cfd:	74 0a                	je     4d09 <jerasure_matrix_decode+0x2b9>
    4cff:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    4d04:	e8 67 c5 ff ff       	callq  1270 <free@plt>
    4d09:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    4d0e:	45 31 e4             	xor    %r12d,%r12d
    4d11:	48 85 c0             	test   %rax,%rax
    4d14:	74 08                	je     4d1e <jerasure_matrix_decode+0x2ce>
    4d16:	48 89 c7             	mov    %rax,%rdi
    4d19:	e8 52 c5 ff ff       	callq  1270 <free@plt>
    4d1e:	48 83 c4 48          	add    $0x48,%rsp
    4d22:	44 89 e0             	mov    %r12d,%eax
    4d25:	5b                   	pop    %rbx
    4d26:	5d                   	pop    %rbp
    4d27:	41 5c                	pop    %r12
    4d29:	41 5d                	pop    %r13
    4d2b:	41 5e                	pop    %r14
    4d2d:	41 5f                	pop    %r15
    4d2f:	c3                   	retq   
    4d30:	41 89 db             	mov    %ebx,%r11d
    4d33:	41 83 fd 01          	cmp    $0x1,%r13d
    4d37:	0f 8e bd fd ff ff    	jle    4afa <jerasure_matrix_decode+0xaa>
    4d3d:	48 63 c3             	movslq %ebx,%rax
    4d40:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    4d45:	48 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%rdi
    4d4c:	00 
    4d4d:	e9 c6 fd ff ff       	jmpq   4b18 <jerasure_matrix_decode+0xc8>
    4d52:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    4d58:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    4d5d:	44 89 5c 24 18       	mov    %r11d,0x18(%rsp)
    4d62:	48 c1 e7 02          	shl    $0x2,%rdi
    4d66:	e8 85 c6 ff ff       	callq  13f0 <malloc@plt>
    4d6b:	85 db                	test   %ebx,%ebx
    4d6d:	44 8b 5c 24 18       	mov    0x18(%rsp),%r11d
    4d72:	48 89 c5             	mov    %rax,%rbp
    4d75:	7e 22                	jle    4d99 <jerasure_matrix_decode+0x349>
    4d77:	8b 4c 24 3c          	mov    0x3c(%rsp),%ecx
    4d7b:	31 d2                	xor    %edx,%edx
    4d7d:	eb 04                	jmp    4d83 <jerasure_matrix_decode+0x333>
    4d7f:	90                   	nop
    4d80:	48 89 c2             	mov    %rax,%rdx
    4d83:	8d 42 01             	lea    0x1(%rdx),%eax
    4d86:	41 39 d3             	cmp    %edx,%r11d
    4d89:	0f 4f c2             	cmovg  %edx,%eax
    4d8c:	89 44 95 00          	mov    %eax,0x0(%rbp,%rdx,4)
    4d90:	48 8d 42 01          	lea    0x1(%rdx),%rax
    4d94:	48 39 ca             	cmp    %rcx,%rdx
    4d97:	75 e7                	jne    4d80 <jerasure_matrix_decode+0x330>
    4d99:	8b 84 24 90 00 00 00 	mov    0x90(%rsp),%eax
    4da0:	45 89 d8             	mov    %r11d,%r8d
    4da3:	48 89 e9             	mov    %rbp,%rcx
    4da6:	44 89 fe             	mov    %r15d,%esi
    4da9:	89 df                	mov    %ebx,%edi
    4dab:	50                   	push   %rax
    4dac:	ff b4 24 90 00 00 00 	pushq  0x90(%rsp)
    4db3:	48 8b 54 24 30       	mov    0x30(%rsp),%rdx
    4db8:	4c 8b 8c 24 90 00 00 	mov    0x90(%rsp),%r9
    4dbf:	00 
    4dc0:	e8 cb f9 ff ff       	callq  4790 <jerasure_matrix_dotprod>
    4dc5:	48 89 ef             	mov    %rbp,%rdi
    4dc8:	e8 a3 c4 ff ff       	callq  1270 <free@plt>
    4dcd:	41 5b                	pop    %r11
    4dcf:	5d                   	pop    %rbp
    4dd0:	e9 75 fe ff ff       	jmpq   4c4a <jerasure_matrix_decode+0x1fa>
    4dd5:	0f 1f 00             	nopl   (%rax)
    4dd8:	45 8b 24 86          	mov    (%r14,%rax,4),%r12d
    4ddc:	45 85 e4             	test   %r12d,%r12d
    4ddf:	0f 85 33 fd ff ff    	jne    4b18 <jerasure_matrix_decode+0xc8>
    4de5:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    4dec:	00 00 
    4dee:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    4df5:	00 00 
    4df7:	45 85 db             	test   %r11d,%r11d
    4dfa:	0f 85 9f fd ff ff    	jne    4b9f <jerasure_matrix_decode+0x14f>
    4e00:	44 89 5c 24 18       	mov    %r11d,0x18(%rsp)
    4e05:	e8 e6 c5 ff ff       	callq  13f0 <malloc@plt>
    4e0a:	44 8b 5c 24 18       	mov    0x18(%rsp),%r11d
    4e0f:	48 89 c5             	mov    %rax,%rbp
    4e12:	e9 60 ff ff ff       	jmpq   4d77 <jerasure_matrix_decode+0x327>
    4e17:	48 63 c3             	movslq %ebx,%rax
    4e1a:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    4e1f:	41 8b 04 86          	mov    (%r14,%rax,4),%eax
    4e23:	85 c0                	test   %eax,%eax
    4e25:	74 55                	je     4e7c <jerasure_matrix_decode+0x42c>
    4e27:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    4e2e:	00 00 
    4e30:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    4e37:	00 00 
    4e39:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    4e40:	00 00 
    4e42:	e9 03 fe ff ff       	jmpq   4c4a <jerasure_matrix_decode+0x1fa>
    4e47:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    4e4e:	00 00 
    4e50:	85 ed                	test   %ebp,%ebp
    4e52:	75 c3                	jne    4e17 <jerasure_matrix_decode+0x3c7>
    4e54:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    4e5b:	00 00 
    4e5d:	8b 7c 24 38          	mov    0x38(%rsp),%edi
    4e61:	48 63 c3             	movslq %ebx,%rax
    4e64:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    4e6b:	00 00 
    4e6d:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    4e72:	85 ff                	test   %edi,%edi
    4e74:	0f 8f e4 fd ff ff    	jg     4c5e <jerasure_matrix_decode+0x20e>
    4e7a:	eb 1e                	jmp    4e9a <jerasure_matrix_decode+0x44a>
    4e7c:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    4e83:	00 00 
    4e85:	8b 74 24 38          	mov    0x38(%rsp),%esi
    4e89:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    4e90:	00 00 
    4e92:	85 f6                	test   %esi,%esi
    4e94:	0f 8f c4 fd ff ff    	jg     4c5e <jerasure_matrix_decode+0x20e>
    4e9a:	4c 89 f7             	mov    %r14,%rdi
    4e9d:	45 31 e4             	xor    %r12d,%r12d
    4ea0:	e8 cb c3 ff ff       	callq  1270 <free@plt>
    4ea5:	e9 74 fe ff ff       	jmpq   4d1e <jerasure_matrix_decode+0x2ce>
    4eaa:	4c 89 f7             	mov    %r14,%rdi
    4ead:	e8 be c3 ff ff       	callq  1270 <free@plt>
    4eb2:	41 bc ff ff ff ff    	mov    $0xffffffff,%r12d
    4eb8:	e9 61 fe ff ff       	jmpq   4d1e <jerasure_matrix_decode+0x2ce>
    4ebd:	4c 89 f7             	mov    %r14,%rdi
    4ec0:	41 bc ff ff ff ff    	mov    $0xffffffff,%r12d
    4ec6:	e8 a5 c3 ff ff       	callq  1270 <free@plt>
    4ecb:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    4ed0:	e8 9b c3 ff ff       	callq  1270 <free@plt>
    4ed5:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    4eda:	e8 91 c3 ff ff       	callq  1270 <free@plt>
    4edf:	e9 3a fe ff ff       	jmpq   4d1e <jerasure_matrix_decode+0x2ce>
    4ee4:	4c 89 f7             	mov    %r14,%rdi
    4ee7:	41 83 cc ff          	or     $0xffffffff,%r12d
    4eeb:	e8 80 c3 ff ff       	callq  1270 <free@plt>
    4ef0:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    4ef5:	e8 76 c3 ff ff       	callq  1270 <free@plt>
    4efa:	e9 1f fe ff ff       	jmpq   4d1e <jerasure_matrix_decode+0x2ce>
    4eff:	90                   	nop

0000000000004f00 <jerasure_matrix_encode>:
    4f00:	f3 0f 1e fa          	endbr64 
    4f04:	41 57                	push   %r15
    4f06:	8d 42 f8             	lea    -0x8(%rdx),%eax
    4f09:	41 89 ff             	mov    %edi,%r15d
    4f0c:	41 56                	push   %r14
    4f0e:	4d 89 ce             	mov    %r9,%r14
    4f11:	41 55                	push   %r13
    4f13:	4d 89 c5             	mov    %r8,%r13
    4f16:	41 54                	push   %r12
    4f18:	55                   	push   %rbp
    4f19:	89 d5                	mov    %edx,%ebp
    4f1b:	53                   	push   %rbx
    4f1c:	48 83 ec 18          	sub    $0x18,%rsp
    4f20:	83 e0 f7             	and    $0xfffffff7,%eax
    4f23:	74 05                	je     4f2a <jerasure_matrix_encode+0x2a>
    4f25:	83 fa 20             	cmp    $0x20,%edx
    4f28:	75 63                	jne    4f8d <jerasure_matrix_encode+0x8d>
    4f2a:	85 f6                	test   %esi,%esi
    4f2c:	7e 50                	jle    4f7e <jerasure_matrix_encode+0x7e>
    4f2e:	4d 63 e7             	movslq %r15d,%r12
    4f31:	48 89 cb             	mov    %rcx,%rbx
    4f34:	4a 8d 04 a5 00 00 00 	lea    0x0(,%r12,4),%rax
    4f3b:	00 
    4f3c:	45 89 fc             	mov    %r15d,%r12d
    4f3f:	48 89 04 24          	mov    %rax,(%rsp)
    4f43:	42 8d 04 3e          	lea    (%rsi,%r15,1),%eax
    4f47:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    4f4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    4f50:	8b 44 24 50          	mov    0x50(%rsp),%eax
    4f54:	48 89 da             	mov    %rbx,%rdx
    4f57:	45 89 e0             	mov    %r12d,%r8d
    4f5a:	4d 89 e9             	mov    %r13,%r9
    4f5d:	31 c9                	xor    %ecx,%ecx
    4f5f:	89 ee                	mov    %ebp,%esi
    4f61:	44 89 ff             	mov    %r15d,%edi
    4f64:	41 83 c4 01          	add    $0x1,%r12d
    4f68:	50                   	push   %rax
    4f69:	41 56                	push   %r14
    4f6b:	e8 20 f8 ff ff       	callq  4790 <jerasure_matrix_dotprod>
    4f70:	48 03 5c 24 10       	add    0x10(%rsp),%rbx
    4f75:	58                   	pop    %rax
    4f76:	5a                   	pop    %rdx
    4f77:	44 3b 64 24 0c       	cmp    0xc(%rsp),%r12d
    4f7c:	75 d2                	jne    4f50 <jerasure_matrix_encode+0x50>
    4f7e:	48 83 c4 18          	add    $0x18,%rsp
    4f82:	5b                   	pop    %rbx
    4f83:	5d                   	pop    %rbp
    4f84:	41 5c                	pop    %r12
    4f86:	41 5d                	pop    %r13
    4f88:	41 5e                	pop    %r14
    4f8a:	41 5f                	pop    %r15
    4f8c:	c3                   	retq   
    4f8d:	48 8b 0d ac c1 00 00 	mov    0xc1ac(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    4f94:	ba 39 00 00 00       	mov    $0x39,%edx
    4f99:	be 01 00 00 00       	mov    $0x1,%esi
    4f9e:	48 8d 3d 03 58 00 00 	lea    0x5803(%rip),%rdi        # a7a8 <__PRETTY_FUNCTION__.5230+0xd7>
    4fa5:	e8 b6 c4 ff ff       	callq  1460 <fwrite@plt>
    4faa:	bf 01 00 00 00       	mov    $0x1,%edi
    4faf:	e8 9c c4 ff ff       	callq  1450 <exit@plt>
    4fb4:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    4fbb:	00 00 00 00 
    4fbf:	90                   	nop

0000000000004fc0 <jerasure_invert_bitmatrix>:
    4fc0:	f3 0f 1e fa          	endbr64 
    4fc4:	41 57                	push   %r15
    4fc6:	41 56                	push   %r14
    4fc8:	41 55                	push   %r13
    4fca:	41 54                	push   %r12
    4fcc:	55                   	push   %rbp
    4fcd:	53                   	push   %rbx
    4fce:	48 89 74 24 c0       	mov    %rsi,-0x40(%rsp)
    4fd3:	8d 72 ff             	lea    -0x1(%rdx),%esi
    4fd6:	48 89 7c 24 b8       	mov    %rdi,-0x48(%rsp)
    4fdb:	89 54 24 e0          	mov    %edx,-0x20(%rsp)
    4fdf:	89 74 24 f4          	mov    %esi,-0xc(%rsp)
    4fe3:	85 d2                	test   %edx,%edx
    4fe5:	0f 8e ef 02 00 00    	jle    52da <jerasure_invert_bitmatrix+0x31a>
    4feb:	48 8b 6c 24 c0       	mov    -0x40(%rsp),%rbp
    4ff0:	44 8b 5c 24 e0       	mov    -0x20(%rsp),%r11d
    4ff5:	31 ff                	xor    %edi,%edi
    4ff7:	31 db                	xor    %ebx,%ebx
    4ff9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    5000:	48 63 c7             	movslq %edi,%rax
    5003:	48 8d 4c 85 00       	lea    0x0(%rbp,%rax,4),%rcx
    5008:	31 c0                	xor    %eax,%eax
    500a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    5010:	31 d2                	xor    %edx,%edx
    5012:	39 c3                	cmp    %eax,%ebx
    5014:	0f 94 c2             	sete   %dl
    5017:	89 14 81             	mov    %edx,(%rcx,%rax,4)
    501a:	48 89 c2             	mov    %rax,%rdx
    501d:	48 83 c0 01          	add    $0x1,%rax
    5021:	48 39 f2             	cmp    %rsi,%rdx
    5024:	75 ea                	jne    5010 <jerasure_invert_bitmatrix+0x50>
    5026:	8d 43 01             	lea    0x1(%rbx),%eax
    5029:	44 01 df             	add    %r11d,%edi
    502c:	41 39 c3             	cmp    %eax,%r11d
    502f:	74 04                	je     5035 <jerasure_invert_bitmatrix+0x75>
    5031:	89 c3                	mov    %eax,%ebx
    5033:	eb cb                	jmp    5000 <jerasure_invert_bitmatrix+0x40>
    5035:	4c 63 64 24 e0       	movslq -0x20(%rsp),%r12
    503a:	89 44 24 f0          	mov    %eax,-0x10(%rsp)
    503e:	89 de                	mov    %ebx,%esi
    5040:	45 31 f6             	xor    %r14d,%r14d
    5043:	48 89 74 24 d0       	mov    %rsi,-0x30(%rsp)
    5048:	45 8d 5e 01          	lea    0x1(%r14),%r11d
    504c:	4a 8d 04 a5 04 00 00 	lea    0x4(,%r12,4),%rax
    5053:	00 
    5054:	c7 44 24 e4 00 00 00 	movl   $0x0,-0x1c(%rsp)
    505b:	00 
    505c:	4a 8d 2c a5 00 00 00 	lea    0x0(,%r12,4),%rbp
    5063:	00 
    5064:	48 89 44 24 e8       	mov    %rax,-0x18(%rsp)
    5069:	48 8b 44 24 b8       	mov    -0x48(%rsp),%rax
    506e:	48 8d 48 04          	lea    0x4(%rax),%rcx
    5072:	48 89 44 24 c8       	mov    %rax,-0x38(%rsp)
    5077:	49 89 c5             	mov    %rax,%r13
    507a:	48 8b 44 24 c8       	mov    -0x38(%rsp),%rax
    507f:	4c 8d 3c b1          	lea    (%rcx,%rsi,4),%r15
    5083:	48 89 4c 24 f8       	mov    %rcx,-0x8(%rsp)
    5088:	31 f6                	xor    %esi,%esi
    508a:	8b 38                	mov    (%rax),%edi
    508c:	85 ff                	test   %edi,%edi
    508e:	0f 84 d4 00 00 00    	je     5168 <jerasure_invert_bitmatrix+0x1a8>
    5094:	0f 1f 40 00          	nopl   0x0(%rax)
    5098:	4c 3b 74 24 d0       	cmp    -0x30(%rsp),%r14
    509d:	0f 84 66 01 00 00    	je     5209 <jerasure_invert_bitmatrix+0x249>
    50a3:	8b 4c 24 e0          	mov    -0x20(%rsp),%ecx
    50a7:	49 8d 04 34          	lea    (%r12,%rsi,1),%rax
    50ab:	01 4c 24 e4          	add    %ecx,-0x1c(%rsp)
    50af:	49 01 ef             	add    %rbp,%r15
    50b2:	4c 89 e1             	mov    %r12,%rcx
    50b5:	48 8d 7c b5 00       	lea    0x0(%rbp,%rsi,4),%rdi
    50ba:	48 89 44 24 d8       	mov    %rax,-0x28(%rsp)
    50bf:	4c 89 fe             	mov    %r15,%rsi
    50c2:	48 f7 d9             	neg    %rcx
    50c5:	eb 1e                	jmp    50e5 <jerasure_invert_bitmatrix+0x125>
    50c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    50ce:	00 00 
    50d0:	41 8d 43 01          	lea    0x1(%r11),%eax
    50d4:	4c 29 e1             	sub    %r12,%rcx
    50d7:	48 01 ee             	add    %rbp,%rsi
    50da:	48 01 ef             	add    %rbp,%rdi
    50dd:	41 39 db             	cmp    %ebx,%r11d
    50e0:	74 4e                	je     5130 <jerasure_invert_bitmatrix+0x170>
    50e2:	41 89 c3             	mov    %eax,%r11d
    50e5:	41 8b 54 3d 00       	mov    0x0(%r13,%rdi,1),%edx
    50ea:	85 d2                	test   %edx,%edx
    50ec:	74 e2                	je     50d0 <jerasure_invert_bitmatrix+0x110>
    50ee:	48 8b 44 24 b8       	mov    -0x48(%rsp),%rax
    50f3:	48 8b 54 24 c0       	mov    -0x40(%rsp),%rdx
    50f8:	48 89 7c 24 b0       	mov    %rdi,-0x50(%rsp)
    50fd:	48 01 f8             	add    %rdi,%rax
    5100:	48 01 fa             	add    %rdi,%rdx
    5103:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5108:	8b 3c 88             	mov    (%rax,%rcx,4),%edi
    510b:	31 38                	xor    %edi,(%rax)
    510d:	48 83 c0 04          	add    $0x4,%rax
    5111:	8b 3c 8a             	mov    (%rdx,%rcx,4),%edi
    5114:	31 3a                	xor    %edi,(%rdx)
    5116:	48 83 c2 04          	add    $0x4,%rdx
    511a:	48 39 f0             	cmp    %rsi,%rax
    511d:	75 e9                	jne    5108 <jerasure_invert_bitmatrix+0x148>
    511f:	48 8b 7c 24 b0       	mov    -0x50(%rsp),%rdi
    5124:	eb aa                	jmp    50d0 <jerasure_invert_bitmatrix+0x110>
    5126:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    512d:	00 00 00 
    5130:	48 8b 4c 24 e8       	mov    -0x18(%rsp),%rcx
    5135:	49 8d 46 01          	lea    0x1(%r14),%rax
    5139:	48 01 4c 24 c8       	add    %rcx,-0x38(%rsp)
    513e:	49 83 c5 04          	add    $0x4,%r13
    5142:	4c 3b 74 24 d0       	cmp    -0x30(%rsp),%r14
    5147:	0f 84 bc 00 00 00    	je     5209 <jerasure_invert_bitmatrix+0x249>
    514d:	49 89 c6             	mov    %rax,%r14
    5150:	48 8b 44 24 c8       	mov    -0x38(%rsp),%rax
    5155:	48 8b 74 24 d8       	mov    -0x28(%rsp),%rsi
    515a:	45 8d 5e 01          	lea    0x1(%r14),%r11d
    515e:	8b 38                	mov    (%rax),%edi
    5160:	85 ff                	test   %edi,%edi
    5162:	0f 85 30 ff ff ff    	jne    5098 <jerasure_invert_bitmatrix+0xd8>
    5168:	44 39 5c 24 f0       	cmp    %r11d,-0x10(%rsp)
    516d:	0f 8e 74 01 00 00    	jle    52e7 <jerasure_invert_bitmatrix+0x327>
    5173:	8b 44 24 e4          	mov    -0x1c(%rsp),%eax
    5177:	03 44 24 e0          	add    -0x20(%rsp),%eax
    517b:	48 8b 4c 24 b8       	mov    -0x48(%rsp),%rcx
    5180:	48 98                	cltq   
    5182:	4c 01 f0             	add    %r14,%rax
    5185:	48 8d 14 81          	lea    (%rcx,%rax,4),%rdx
    5189:	44 89 d8             	mov    %r11d,%eax
    518c:	eb 12                	jmp    51a0 <jerasure_invert_bitmatrix+0x1e0>
    518e:	66 90                	xchg   %ax,%ax
    5190:	8d 48 01             	lea    0x1(%rax),%ecx
    5193:	48 01 ea             	add    %rbp,%rdx
    5196:	39 d8                	cmp    %ebx,%eax
    5198:	0f 84 08 01 00 00    	je     52a6 <jerasure_invert_bitmatrix+0x2e6>
    519e:	89 c8                	mov    %ecx,%eax
    51a0:	8b 0a                	mov    (%rdx),%ecx
    51a2:	85 c9                	test   %ecx,%ecx
    51a4:	74 ea                	je     5190 <jerasure_invert_bitmatrix+0x1d0>
    51a6:	8b 4c 24 f0          	mov    -0x10(%rsp),%ecx
    51aa:	39 c1                	cmp    %eax,%ecx
    51ac:	0f 84 f4 00 00 00    	je     52a6 <jerasure_invert_bitmatrix+0x2e6>
    51b2:	0f af c1             	imul   %ecx,%eax
    51b5:	48 8b 54 24 b8       	mov    -0x48(%rsp),%rdx
    51ba:	48 89 74 24 b0       	mov    %rsi,-0x50(%rsp)
    51bf:	48 8d 0c b5 00 00 00 	lea    0x0(,%rsi,4),%rcx
    51c6:	00 
    51c7:	48 01 ca             	add    %rcx,%rdx
    51ca:	48 03 4c 24 c0       	add    -0x40(%rsp),%rcx
    51cf:	48 98                	cltq   
    51d1:	48 29 f0             	sub    %rsi,%rax
    51d4:	0f 1f 40 00          	nopl   0x0(%rax)
    51d8:	8b 3a                	mov    (%rdx),%edi
    51da:	8b 34 82             	mov    (%rdx,%rax,4),%esi
    51dd:	89 32                	mov    %esi,(%rdx)
    51df:	89 3c 82             	mov    %edi,(%rdx,%rax,4)
    51e2:	8b 39                	mov    (%rcx),%edi
    51e4:	48 83 c2 04          	add    $0x4,%rdx
    51e8:	8b 34 81             	mov    (%rcx,%rax,4),%esi
    51eb:	89 31                	mov    %esi,(%rcx)
    51ed:	89 3c 81             	mov    %edi,(%rcx,%rax,4)
    51f0:	48 83 c1 04          	add    $0x4,%rcx
    51f4:	4c 39 fa             	cmp    %r15,%rdx
    51f7:	75 df                	jne    51d8 <jerasure_invert_bitmatrix+0x218>
    51f9:	48 8b 74 24 b0       	mov    -0x50(%rsp),%rsi
    51fe:	4c 3b 74 24 d0       	cmp    -0x30(%rsp),%r14
    5203:	0f 85 9a fe ff ff    	jne    50a3 <jerasure_invert_bitmatrix+0xe3>
    5209:	48 8b 74 24 b8       	mov    -0x48(%rsp),%rsi
    520e:	48 63 c3             	movslq %ebx,%rax
    5211:	44 8b 74 24 e0       	mov    -0x20(%rsp),%r14d
    5216:	4a 8d 2c a5 00 00 00 	lea    0x0(,%r12,4),%rbp
    521d:	00 
    521e:	44 0f af f3          	imul   %ebx,%r14d
    5222:	4c 8d 2c 86          	lea    (%rsi,%rax,4),%r13
    5226:	8b 44 24 f4          	mov    -0xc(%rsp),%eax
    522a:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
    522f:	4c 8d 3c 86          	lea    (%rsi,%rax,4),%r15
    5233:	4c 89 7c 24 c8       	mov    %r15,-0x38(%rsp)
    5238:	4c 8b 7c 24 c0       	mov    -0x40(%rsp),%r15
    523d:	85 db                	test   %ebx,%ebx
    523f:	0f 8e 90 00 00 00    	jle    52d5 <jerasure_invert_bitmatrix+0x315>
    5245:	48 8b 74 24 c8       	mov    -0x38(%rsp),%rsi
    524a:	49 63 ce             	movslq %r14d,%rcx
    524d:	31 ff                	xor    %edi,%edi
    524f:	45 31 db             	xor    %r11d,%r11d
    5252:	eb 16                	jmp    526a <jerasure_invert_bitmatrix+0x2aa>
    5254:	0f 1f 40 00          	nopl   0x0(%rax)
    5258:	41 83 c3 01          	add    $0x1,%r11d
    525c:	4c 29 e1             	sub    %r12,%rcx
    525f:	48 01 ee             	add    %rbp,%rsi
    5262:	48 01 ef             	add    %rbp,%rdi
    5265:	41 39 db             	cmp    %ebx,%r11d
    5268:	74 56                	je     52c0 <jerasure_invert_bitmatrix+0x300>
    526a:	41 8b 44 3d 00       	mov    0x0(%r13,%rdi,1),%eax
    526f:	85 c0                	test   %eax,%eax
    5271:	74 e5                	je     5258 <jerasure_invert_bitmatrix+0x298>
    5273:	48 8b 44 24 b8       	mov    -0x48(%rsp),%rax
    5278:	48 89 7c 24 b0       	mov    %rdi,-0x50(%rsp)
    527d:	49 8d 14 3f          	lea    (%r15,%rdi,1),%rdx
    5281:	48 01 f8             	add    %rdi,%rax
    5284:	0f 1f 40 00          	nopl   0x0(%rax)
    5288:	8b 3c 88             	mov    (%rax,%rcx,4),%edi
    528b:	31 38                	xor    %edi,(%rax)
    528d:	48 83 c0 04          	add    $0x4,%rax
    5291:	8b 3c 8a             	mov    (%rdx,%rcx,4),%edi
    5294:	31 3a                	xor    %edi,(%rdx)
    5296:	48 83 c2 04          	add    $0x4,%rdx
    529a:	48 39 c6             	cmp    %rax,%rsi
    529d:	75 e9                	jne    5288 <jerasure_invert_bitmatrix+0x2c8>
    529f:	48 8b 7c 24 b0       	mov    -0x50(%rsp),%rdi
    52a4:	eb b2                	jmp    5258 <jerasure_invert_bitmatrix+0x298>
    52a6:	5b                   	pop    %rbx
    52a7:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    52ac:	5d                   	pop    %rbp
    52ad:	41 5c                	pop    %r12
    52af:	41 5d                	pop    %r13
    52b1:	41 5e                	pop    %r14
    52b3:	41 5f                	pop    %r15
    52b5:	c3                   	retq   
    52b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    52bd:	00 00 00 
    52c0:	41 8d 5b ff          	lea    -0x1(%r11),%ebx
    52c4:	49 83 ed 04          	sub    $0x4,%r13
    52c8:	44 2b 74 24 e0       	sub    -0x20(%rsp),%r14d
    52cd:	85 db                	test   %ebx,%ebx
    52cf:	0f 8f 70 ff ff ff    	jg     5245 <jerasure_invert_bitmatrix+0x285>
    52d5:	83 eb 01             	sub    $0x1,%ebx
    52d8:	79 ea                	jns    52c4 <jerasure_invert_bitmatrix+0x304>
    52da:	5b                   	pop    %rbx
    52db:	31 c0                	xor    %eax,%eax
    52dd:	5d                   	pop    %rbp
    52de:	41 5c                	pop    %r12
    52e0:	41 5d                	pop    %r13
    52e2:	41 5e                	pop    %r14
    52e4:	41 5f                	pop    %r15
    52e6:	c3                   	retq   
    52e7:	44 89 d8             	mov    %r11d,%eax
    52ea:	e9 b7 fe ff ff       	jmpq   51a6 <jerasure_invert_bitmatrix+0x1e6>
    52ef:	90                   	nop

00000000000052f0 <jerasure_make_decoding_bitmatrix>:
    52f0:	f3 0f 1e fa          	endbr64 
    52f4:	89 f8                	mov    %edi,%eax
    52f6:	41 57                	push   %r15
    52f8:	0f af c2             	imul   %edx,%eax
    52fb:	41 56                	push   %r14
    52fd:	41 55                	push   %r13
    52ff:	41 54                	push   %r12
    5301:	0f af c0             	imul   %eax,%eax
    5304:	55                   	push   %rbp
    5305:	89 fd                	mov    %edi,%ebp
    5307:	53                   	push   %rbx
    5308:	89 d3                	mov    %edx,%ebx
    530a:	48 63 f8             	movslq %eax,%rdi
    530d:	48 83 ec 38          	sub    $0x38,%rsp
    5311:	48 c1 e7 02          	shl    $0x2,%rdi
    5315:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    531a:	4c 8b 64 24 70       	mov    0x70(%rsp),%r12
    531f:	4c 89 4c 24 28       	mov    %r9,0x28(%rsp)
    5324:	85 ed                	test   %ebp,%ebp
    5326:	0f 8e 5c 01 00 00    	jle    5488 <jerasure_make_decoding_bitmatrix+0x198>
    532c:	4c 89 c1             	mov    %r8,%rcx
    532f:	31 c0                	xor    %eax,%eax
    5331:	31 d2                	xor    %edx,%edx
    5333:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5338:	8b 34 81             	mov    (%rcx,%rax,4),%esi
    533b:	85 f6                	test   %esi,%esi
    533d:	75 0a                	jne    5349 <jerasure_make_decoding_bitmatrix+0x59>
    533f:	48 63 f2             	movslq %edx,%rsi
    5342:	83 c2 01             	add    $0x1,%edx
    5345:	41 89 04 b4          	mov    %eax,(%r12,%rsi,4)
    5349:	48 83 c0 01          	add    $0x1,%rax
    534d:	39 ea                	cmp    %ebp,%edx
    534f:	7c e7                	jl     5338 <jerasure_make_decoding_bitmatrix+0x48>
    5351:	e8 9a c0 ff ff       	callq  13f0 <malloc@plt>
    5356:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    535b:	48 85 c0             	test   %rax,%rax
    535e:	0f 84 33 01 00 00    	je     5497 <jerasure_make_decoding_bitmatrix+0x1a7>
    5364:	89 d8                	mov    %ebx,%eax
    5366:	41 89 df             	mov    %ebx,%r15d
    5369:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    536e:	4d 89 e3             	mov    %r12,%r11
    5371:	0f af c5             	imul   %ebp,%eax
    5374:	45 31 f6             	xor    %r14d,%r14d
    5377:	44 0f af f8          	imul   %eax,%r15d
    537b:	49 63 d7             	movslq %r15d,%rdx
    537e:	45 8d 6f ff          	lea    -0x1(%r15),%r13d
    5382:	48 8d 34 95 00 00 00 	lea    0x0(,%rdx,4),%rsi
    5389:	00 
    538a:	8d 55 ff             	lea    -0x1(%rbp),%edx
    538d:	49 8d 54 94 04       	lea    0x4(%r12,%rdx,4),%rdx
    5392:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
    5397:	48 89 fe             	mov    %rdi,%rsi
    539a:	4a 8d 7c af 04       	lea    0x4(%rdi,%r13,4),%rdi
    539f:	48 89 54 24 10       	mov    %rdx,0x10(%rsp)
    53a4:	8d 50 01             	lea    0x1(%rax),%edx
    53a7:	48 63 d2             	movslq %edx,%rdx
    53aa:	48 c1 e2 02          	shl    $0x2,%rdx
    53ae:	66 90                	xchg   %ax,%ax
    53b0:	41 8b 0b             	mov    (%r11),%ecx
    53b3:	39 e9                	cmp    %ebp,%ecx
    53b5:	0f 8d 95 00 00 00    	jge    5450 <jerasure_make_decoding_bitmatrix+0x160>
    53bb:	48 89 f0             	mov    %rsi,%rax
    53be:	45 85 ff             	test   %r15d,%r15d
    53c1:	7e 14                	jle    53d7 <jerasure_make_decoding_bitmatrix+0xe7>
    53c3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    53c8:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
    53ce:	48 83 c0 04          	add    $0x4,%rax
    53d2:	48 39 c7             	cmp    %rax,%rdi
    53d5:	75 f1                	jne    53c8 <jerasure_make_decoding_bitmatrix+0xd8>
    53d7:	0f af cb             	imul   %ebx,%ecx
    53da:	44 01 f1             	add    %r14d,%ecx
    53dd:	85 db                	test   %ebx,%ebx
    53df:	7e 1f                	jle    5400 <jerasure_make_decoding_bitmatrix+0x110>
    53e1:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    53e6:	48 63 c9             	movslq %ecx,%rcx
    53e9:	48 8d 0c 88          	lea    (%rax,%rcx,4),%rcx
    53ed:	31 c0                	xor    %eax,%eax
    53ef:	90                   	nop
    53f0:	83 c0 01             	add    $0x1,%eax
    53f3:	c7 01 01 00 00 00    	movl   $0x1,(%rcx)
    53f9:	48 01 d1             	add    %rdx,%rcx
    53fc:	39 c3                	cmp    %eax,%ebx
    53fe:	75 f0                	jne    53f0 <jerasure_make_decoding_bitmatrix+0x100>
    5400:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    5405:	49 83 c3 04          	add    $0x4,%r11
    5409:	45 01 fe             	add    %r15d,%r14d
    540c:	48 01 c6             	add    %rax,%rsi
    540f:	48 01 c7             	add    %rax,%rdi
    5412:	4c 39 5c 24 10       	cmp    %r11,0x10(%rsp)
    5417:	75 97                	jne    53b0 <jerasure_make_decoding_bitmatrix+0xc0>
    5419:	89 ea                	mov    %ebp,%edx
    541b:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
    5420:	0f af d3             	imul   %ebx,%edx
    5423:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
    5428:	48 89 df             	mov    %rbx,%rdi
    542b:	e8 90 fb ff ff       	callq  4fc0 <jerasure_invert_bitmatrix>
    5430:	48 89 df             	mov    %rbx,%rdi
    5433:	41 89 c4             	mov    %eax,%r12d
    5436:	e8 35 be ff ff       	callq  1270 <free@plt>
    543b:	48 83 c4 38          	add    $0x38,%rsp
    543f:	44 89 e0             	mov    %r12d,%eax
    5442:	5b                   	pop    %rbx
    5443:	5d                   	pop    %rbp
    5444:	41 5c                	pop    %r12
    5446:	41 5d                	pop    %r13
    5448:	41 5e                	pop    %r14
    544a:	41 5f                	pop    %r15
    544c:	c3                   	retq   
    544d:	0f 1f 00             	nopl   (%rax)
    5450:	29 e9                	sub    %ebp,%ecx
    5452:	0f af cd             	imul   %ebp,%ecx
    5455:	0f af cb             	imul   %ebx,%ecx
    5458:	0f af cb             	imul   %ebx,%ecx
    545b:	45 85 ff             	test   %r15d,%r15d
    545e:	7e a0                	jle    5400 <jerasure_make_decoding_bitmatrix+0x110>
    5460:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    5465:	48 63 c9             	movslq %ecx,%rcx
    5468:	4c 8d 24 88          	lea    (%rax,%rcx,4),%r12
    546c:	31 c0                	xor    %eax,%eax
    546e:	66 90                	xchg   %ax,%ax
    5470:	41 8b 0c 84          	mov    (%r12,%rax,4),%ecx
    5474:	89 0c 86             	mov    %ecx,(%rsi,%rax,4)
    5477:	48 89 c1             	mov    %rax,%rcx
    547a:	48 83 c0 01          	add    $0x1,%rax
    547e:	4c 39 e9             	cmp    %r13,%rcx
    5481:	75 ed                	jne    5470 <jerasure_make_decoding_bitmatrix+0x180>
    5483:	e9 78 ff ff ff       	jmpq   5400 <jerasure_make_decoding_bitmatrix+0x110>
    5488:	e8 63 bf ff ff       	callq  13f0 <malloc@plt>
    548d:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    5492:	48 85 c0             	test   %rax,%rax
    5495:	75 82                	jne    5419 <jerasure_make_decoding_bitmatrix+0x129>
    5497:	41 83 cc ff          	or     $0xffffffff,%r12d
    549b:	eb 9e                	jmp    543b <jerasure_make_decoding_bitmatrix+0x14b>
    549d:	0f 1f 00             	nopl   (%rax)

00000000000054a0 <jerasure_bitmatrix_decode>:
    54a0:	f3 0f 1e fa          	endbr64 
    54a4:	41 57                	push   %r15
    54a6:	41 89 d7             	mov    %edx,%r15d
    54a9:	4c 89 ca             	mov    %r9,%rdx
    54ac:	41 56                	push   %r14
    54ae:	41 55                	push   %r13
    54b0:	41 89 fd             	mov    %edi,%r13d
    54b3:	41 54                	push   %r12
    54b5:	55                   	push   %rbp
    54b6:	53                   	push   %rbx
    54b7:	44 89 c3             	mov    %r8d,%ebx
    54ba:	48 83 ec 48          	sub    $0x48,%rsp
    54be:	89 74 24 38          	mov    %esi,0x38(%rsp)
    54c2:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
    54c7:	e8 d4 ef ff ff       	callq  44a0 <jerasure_erasures_to_erased>
    54cc:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    54d1:	48 85 c0             	test   %rax,%rax
    54d4:	0f 84 6b 04 00 00    	je     5945 <jerasure_bitmatrix_decode+0x4a5>
    54da:	45 85 ed             	test   %r13d,%r13d
    54dd:	0f 8e ed 03 00 00    	jle    58d0 <jerasure_bitmatrix_decode+0x430>
    54e3:	41 8d 4d ff          	lea    -0x1(%r13),%ecx
    54e7:	48 89 c6             	mov    %rax,%rsi
    54ea:	45 89 eb             	mov    %r13d,%r11d
    54ed:	31 c0                	xor    %eax,%eax
    54ef:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
    54f3:	45 31 e4             	xor    %r12d,%r12d
    54f6:	eb 0b                	jmp    5503 <jerasure_bitmatrix_decode+0x63>
    54f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    54ff:	00 
    5500:	48 89 d0             	mov    %rdx,%rax
    5503:	8b 3c 86             	mov    (%rsi,%rax,4),%edi
    5506:	85 ff                	test   %edi,%edi
    5508:	74 07                	je     5511 <jerasure_bitmatrix_decode+0x71>
    550a:	41 83 c4 01          	add    $0x1,%r12d
    550e:	41 89 c3             	mov    %eax,%r11d
    5511:	48 8d 50 01          	lea    0x1(%rax),%rdx
    5515:	48 39 c8             	cmp    %rcx,%rax
    5518:	75 e6                	jne    5500 <jerasure_bitmatrix_decode+0x60>
    551a:	83 fb 01             	cmp    $0x1,%ebx
    551d:	0f 84 25 03 00 00    	je     5848 <jerasure_bitmatrix_decode+0x3a8>
    5523:	45 89 eb             	mov    %r13d,%r11d
    5526:	41 83 fc 01          	cmp    $0x1,%r12d
    552a:	0f 8f 00 03 00 00    	jg     5830 <jerasure_bitmatrix_decode+0x390>
    5530:	0f 85 2a 03 00 00    	jne    5860 <jerasure_bitmatrix_decode+0x3c0>
    5536:	49 63 c5             	movslq %r13d,%rax
    5539:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    553e:	48 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%rdi
    5545:	00 
    5546:	83 fb 01             	cmp    $0x1,%ebx
    5549:	0f 84 31 03 00 00    	je     5880 <jerasure_bitmatrix_decode+0x3e0>
    554f:	44 89 5c 24 10       	mov    %r11d,0x10(%rsp)
    5554:	e8 97 be ff ff       	callq  13f0 <malloc@plt>
    5559:	44 8b 5c 24 10       	mov    0x10(%rsp),%r11d
    555e:	48 85 c0             	test   %rax,%rax
    5561:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    5566:	0f 84 2a 04 00 00    	je     5996 <jerasure_bitmatrix_decode+0x4f6>
    556c:	44 89 ef             	mov    %r13d,%edi
    556f:	44 89 5c 24 18       	mov    %r11d,0x18(%rsp)
    5574:	41 0f af ff          	imul   %r15d,%edi
    5578:	0f af ff             	imul   %edi,%edi
    557b:	48 63 ff             	movslq %edi,%rdi
    557e:	48 c1 e7 02          	shl    $0x2,%rdi
    5582:	e8 69 be ff ff       	callq  13f0 <malloc@plt>
    5587:	44 8b 5c 24 18       	mov    0x18(%rsp),%r11d
    558c:	48 85 c0             	test   %rax,%rax
    558f:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    5594:	0f 84 df 03 00 00    	je     5979 <jerasure_bitmatrix_decode+0x4d9>
    559a:	44 89 5c 24 18       	mov    %r11d,0x18(%rsp)
    559f:	48 83 ec 08          	sub    $0x8,%rsp
    55a3:	44 89 fa             	mov    %r15d,%edx
    55a6:	44 89 ef             	mov    %r13d,%edi
    55a9:	ff 74 24 10          	pushq  0x10(%rsp)
    55ad:	4c 8b 4c 24 20       	mov    0x20(%rsp),%r9
    55b2:	4c 8b 44 24 30       	mov    0x30(%rsp),%r8
    55b7:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    55bc:	8b 74 24 48          	mov    0x48(%rsp),%esi
    55c0:	e8 2b fd ff ff       	callq  52f0 <jerasure_make_decoding_bitmatrix>
    55c5:	41 5e                	pop    %r14
    55c7:	5a                   	pop    %rdx
    55c8:	85 c0                	test   %eax,%eax
    55ca:	0f 88 80 03 00 00    	js     5950 <jerasure_bitmatrix_decode+0x4b0>
    55d0:	44 8b 5c 24 18       	mov    0x18(%rsp),%r11d
    55d5:	45 85 db             	test   %r11d,%r11d
    55d8:	0f 8e bb 00 00 00    	jle    5699 <jerasure_bitmatrix_decode+0x1f9>
    55de:	44 89 f8             	mov    %r15d,%eax
    55e1:	44 89 ef             	mov    %r13d,%edi
    55e4:	44 89 fe             	mov    %r15d,%esi
    55e7:	48 8b 6c 24 20       	mov    0x20(%rsp),%rbp
    55ec:	41 0f af c7          	imul   %r15d,%eax
    55f0:	45 31 f6             	xor    %r14d,%r14d
    55f3:	31 db                	xor    %ebx,%ebx
    55f5:	41 0f af c5          	imul   %r13d,%eax
    55f9:	45 89 dd             	mov    %r11d,%r13d
    55fc:	41 89 c7             	mov    %eax,%r15d
    55ff:	eb 1b                	jmp    561c <jerasure_bitmatrix_decode+0x17c>
    5601:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    5608:	83 c3 01             	add    $0x1,%ebx
    560b:	48 83 c5 04          	add    $0x4,%rbp
    560f:	45 01 fe             	add    %r15d,%r14d
    5612:	45 85 e4             	test   %r12d,%r12d
    5615:	7e 79                	jle    5690 <jerasure_bitmatrix_decode+0x1f0>
    5617:	41 39 dd             	cmp    %ebx,%r13d
    561a:	7e 74                	jle    5690 <jerasure_bitmatrix_decode+0x1f0>
    561c:	44 8b 5d 00          	mov    0x0(%rbp),%r11d
    5620:	45 85 db             	test   %r11d,%r11d
    5623:	74 e3                	je     5608 <jerasure_bitmatrix_decode+0x168>
    5625:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    562a:	49 63 d6             	movslq %r14d,%rdx
    562d:	48 83 ec 08          	sub    $0x8,%rsp
    5631:	41 89 d8             	mov    %ebx,%r8d
    5634:	41 83 ec 01          	sub    $0x1,%r12d
    5638:	83 c3 01             	add    $0x1,%ebx
    563b:	48 83 c5 04          	add    $0x4,%rbp
    563f:	45 01 fe             	add    %r15d,%r14d
    5642:	48 8d 14 90          	lea    (%rax,%rdx,4),%rdx
    5646:	8b 84 24 a0 00 00 00 	mov    0xa0(%rsp),%eax
    564d:	50                   	push   %rax
    564e:	8b 84 24 a0 00 00 00 	mov    0xa0(%rsp),%eax
    5655:	50                   	push   %rax
    5656:	ff b4 24 a0 00 00 00 	pushq  0xa0(%rsp)
    565d:	4c 8b 8c 24 a0 00 00 	mov    0xa0(%rsp),%r9
    5664:	00 
    5665:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
    566a:	89 74 24 3c          	mov    %esi,0x3c(%rsp)
    566e:	89 7c 24 38          	mov    %edi,0x38(%rsp)
    5672:	e8 09 e1 ff ff       	callq  3780 <jerasure_bitmatrix_dotprod>
    5677:	48 83 c4 20          	add    $0x20,%rsp
    567b:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
    567f:	8b 7c 24 18          	mov    0x18(%rsp),%edi
    5683:	45 85 e4             	test   %r12d,%r12d
    5686:	7f 8f                	jg     5617 <jerasure_bitmatrix_decode+0x177>
    5688:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    568f:	00 
    5690:	45 89 eb             	mov    %r13d,%r11d
    5693:	41 89 f7             	mov    %esi,%r15d
    5696:	41 89 fd             	mov    %edi,%r13d
    5699:	44 89 5c 24 18       	mov    %r11d,0x18(%rsp)
    569e:	45 85 e4             	test   %r12d,%r12d
    56a1:	0f 85 f9 00 00 00    	jne    57a0 <jerasure_bitmatrix_decode+0x300>
    56a7:	8b 4c 24 38          	mov    0x38(%rsp),%ecx
    56ab:	49 63 c5             	movslq %r13d,%rax
    56ae:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    56b3:	85 c9                	test   %ecx,%ecx
    56b5:	0f 8e 9e 00 00 00    	jle    5759 <jerasure_bitmatrix_decode+0x2b9>
    56bb:	45 89 fe             	mov    %r15d,%r14d
    56be:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    56c3:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    56c8:	44 89 eb             	mov    %r13d,%ebx
    56cb:	45 0f af f7          	imul   %r15d,%r14d
    56cf:	44 89 7c 24 18       	mov    %r15d,0x18(%rsp)
    56d4:	45 31 e4             	xor    %r12d,%r12d
    56d7:	48 8d 2c 88          	lea    (%rax,%rcx,4),%rbp
    56db:	8b 44 24 38          	mov    0x38(%rsp),%eax
    56df:	45 0f af f5          	imul   %r13d,%r14d
    56e3:	44 01 e8             	add    %r13d,%eax
    56e6:	45 89 f7             	mov    %r14d,%r15d
    56e9:	41 89 c6             	mov    %eax,%r14d
    56ec:	eb 11                	jmp    56ff <jerasure_bitmatrix_decode+0x25f>
    56ee:	66 90                	xchg   %ax,%ax
    56f0:	83 c3 01             	add    $0x1,%ebx
    56f3:	48 83 c5 04          	add    $0x4,%rbp
    56f7:	45 01 fc             	add    %r15d,%r12d
    56fa:	41 39 de             	cmp    %ebx,%r14d
    56fd:	74 5a                	je     5759 <jerasure_bitmatrix_decode+0x2b9>
    56ff:	8b 55 00             	mov    0x0(%rbp),%edx
    5702:	85 d2                	test   %edx,%edx
    5704:	74 ea                	je     56f0 <jerasure_bitmatrix_decode+0x250>
    5706:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    570b:	49 63 d4             	movslq %r12d,%rdx
    570e:	48 83 ec 08          	sub    $0x8,%rsp
    5712:	41 89 d8             	mov    %ebx,%r8d
    5715:	31 c9                	xor    %ecx,%ecx
    5717:	44 89 ef             	mov    %r13d,%edi
    571a:	83 c3 01             	add    $0x1,%ebx
    571d:	48 83 c5 04          	add    $0x4,%rbp
    5721:	48 8d 14 90          	lea    (%rax,%rdx,4),%rdx
    5725:	8b 84 24 a0 00 00 00 	mov    0xa0(%rsp),%eax
    572c:	45 01 fc             	add    %r15d,%r12d
    572f:	50                   	push   %rax
    5730:	8b 84 24 a0 00 00 00 	mov    0xa0(%rsp),%eax
    5737:	50                   	push   %rax
    5738:	ff b4 24 a0 00 00 00 	pushq  0xa0(%rsp)
    573f:	4c 8b 8c 24 a0 00 00 	mov    0xa0(%rsp),%r9
    5746:	00 
    5747:	8b 74 24 38          	mov    0x38(%rsp),%esi
    574b:	e8 30 e0 ff ff       	callq  3780 <jerasure_bitmatrix_dotprod>
    5750:	48 83 c4 20          	add    $0x20,%rsp
    5754:	41 39 de             	cmp    %ebx,%r14d
    5757:	75 a6                	jne    56ff <jerasure_bitmatrix_decode+0x25f>
    5759:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    575e:	e8 0d bb ff ff       	callq  1270 <free@plt>
    5763:	48 83 7c 24 08 00    	cmpq   $0x0,0x8(%rsp)
    5769:	74 0a                	je     5775 <jerasure_bitmatrix_decode+0x2d5>
    576b:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    5770:	e8 fb ba ff ff       	callq  1270 <free@plt>
    5775:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    577a:	45 31 e4             	xor    %r12d,%r12d
    577d:	48 85 ff             	test   %rdi,%rdi
    5780:	74 05                	je     5787 <jerasure_bitmatrix_decode+0x2e7>
    5782:	e8 e9 ba ff ff       	callq  1270 <free@plt>
    5787:	48 83 c4 48          	add    $0x48,%rsp
    578b:	44 89 e0             	mov    %r12d,%eax
    578e:	5b                   	pop    %rbx
    578f:	5d                   	pop    %rbp
    5790:	41 5c                	pop    %r12
    5792:	41 5d                	pop    %r13
    5794:	41 5e                	pop    %r14
    5796:	41 5f                	pop    %r15
    5798:	c3                   	retq   
    5799:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    57a0:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    57a5:	48 c1 e7 02          	shl    $0x2,%rdi
    57a9:	e8 42 bc ff ff       	callq  13f0 <malloc@plt>
    57ae:	45 85 ed             	test   %r13d,%r13d
    57b1:	44 8b 5c 24 18       	mov    0x18(%rsp),%r11d
    57b6:	48 89 c5             	mov    %rax,%rbp
    57b9:	7e 26                	jle    57e1 <jerasure_bitmatrix_decode+0x341>
    57bb:	8b 4c 24 3c          	mov    0x3c(%rsp),%ecx
    57bf:	31 d2                	xor    %edx,%edx
    57c1:	eb 08                	jmp    57cb <jerasure_bitmatrix_decode+0x32b>
    57c3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    57c8:	48 89 c2             	mov    %rax,%rdx
    57cb:	8d 42 01             	lea    0x1(%rdx),%eax
    57ce:	41 39 d3             	cmp    %edx,%r11d
    57d1:	0f 4f c2             	cmovg  %edx,%eax
    57d4:	89 44 95 00          	mov    %eax,0x0(%rbp,%rdx,4)
    57d8:	48 8d 42 01          	lea    0x1(%rdx),%rax
    57dc:	48 39 ca             	cmp    %rcx,%rdx
    57df:	75 e7                	jne    57c8 <jerasure_bitmatrix_decode+0x328>
    57e1:	48 83 ec 08          	sub    $0x8,%rsp
    57e5:	44 89 ef             	mov    %r13d,%edi
    57e8:	45 89 d8             	mov    %r11d,%r8d
    57eb:	48 89 e9             	mov    %rbp,%rcx
    57ee:	8b 84 24 a0 00 00 00 	mov    0xa0(%rsp),%eax
    57f5:	44 89 fe             	mov    %r15d,%esi
    57f8:	50                   	push   %rax
    57f9:	8b 84 24 a0 00 00 00 	mov    0xa0(%rsp),%eax
    5800:	50                   	push   %rax
    5801:	ff b4 24 a0 00 00 00 	pushq  0xa0(%rsp)
    5808:	48 8b 54 24 48       	mov    0x48(%rsp),%rdx
    580d:	4c 8b 8c 24 a0 00 00 	mov    0xa0(%rsp),%r9
    5814:	00 
    5815:	e8 66 df ff ff       	callq  3780 <jerasure_bitmatrix_dotprod>
    581a:	48 83 c4 20          	add    $0x20,%rsp
    581e:	48 89 ef             	mov    %rbp,%rdi
    5821:	e8 4a ba ff ff       	callq  1270 <free@plt>
    5826:	e9 7c fe ff ff       	jmpq   56a7 <jerasure_bitmatrix_decode+0x207>
    582b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5830:	49 63 c5             	movslq %r13d,%rax
    5833:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    5838:	48 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%rdi
    583f:	00 
    5840:	e9 0a fd ff ff       	jmpq   554f <jerasure_bitmatrix_decode+0xaf>
    5845:	0f 1f 00             	nopl   (%rax)
    5848:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    584d:	49 63 c5             	movslq %r13d,%rax
    5850:	8b 34 81             	mov    (%rcx,%rax,4),%esi
    5853:	85 f6                	test   %esi,%esi
    5855:	45 0f 45 dd          	cmovne %r13d,%r11d
    5859:	e9 c8 fc ff ff       	jmpq   5526 <jerasure_bitmatrix_decode+0x86>
    585e:	66 90                	xchg   %ax,%ax
    5860:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    5867:	00 00 
    5869:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    5870:	00 00 
    5872:	e9 30 fe ff ff       	jmpq   56a7 <jerasure_bitmatrix_decode+0x207>
    5877:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    587e:	00 00 
    5880:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    5885:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    588a:	8b 0c 88             	mov    (%rax,%rcx,4),%ecx
    588d:	85 c9                	test   %ecx,%ecx
    588f:	0f 85 ba fc ff ff    	jne    554f <jerasure_bitmatrix_decode+0xaf>
    5895:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    589c:	00 00 
    589e:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    58a5:	00 00 
    58a7:	45 85 db             	test   %r11d,%r11d
    58aa:	0f 85 2e fd ff ff    	jne    55de <jerasure_bitmatrix_decode+0x13e>
    58b0:	44 89 5c 24 18       	mov    %r11d,0x18(%rsp)
    58b5:	e8 36 bb ff ff       	callq  13f0 <malloc@plt>
    58ba:	44 8b 5c 24 18       	mov    0x18(%rsp),%r11d
    58bf:	48 89 c5             	mov    %rax,%rbp
    58c2:	e9 f4 fe ff ff       	jmpq   57bb <jerasure_bitmatrix_decode+0x31b>
    58c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    58ce:	00 00 
    58d0:	83 fb 01             	cmp    $0x1,%ebx
    58d3:	74 38                	je     590d <jerasure_bitmatrix_decode+0x46d>
    58d5:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    58dc:	00 00 
    58de:	8b 6c 24 38          	mov    0x38(%rsp),%ebp
    58e2:	49 63 c5             	movslq %r13d,%rax
    58e5:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    58ec:	00 00 
    58ee:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    58f3:	85 ed                	test   %ebp,%ebp
    58f5:	0f 8f c0 fd ff ff    	jg     56bb <jerasure_bitmatrix_decode+0x21b>
    58fb:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    5900:	45 31 e4             	xor    %r12d,%r12d
    5903:	e8 68 b9 ff ff       	callq  1270 <free@plt>
    5908:	e9 7a fe ff ff       	jmpq   5787 <jerasure_bitmatrix_decode+0x2e7>
    590d:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    5912:	49 63 c5             	movslq %r13d,%rax
    5915:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    591a:	8b 04 81             	mov    (%rcx,%rax,4),%eax
    591d:	85 c0                	test   %eax,%eax
    591f:	0f 85 3b ff ff ff    	jne    5860 <jerasure_bitmatrix_decode+0x3c0>
    5925:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    592c:	00 00 
    592e:	8b 5c 24 38          	mov    0x38(%rsp),%ebx
    5932:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    5939:	00 00 
    593b:	85 db                	test   %ebx,%ebx
    593d:	0f 8f 78 fd ff ff    	jg     56bb <jerasure_bitmatrix_decode+0x21b>
    5943:	eb b6                	jmp    58fb <jerasure_bitmatrix_decode+0x45b>
    5945:	41 bc ff ff ff ff    	mov    $0xffffffff,%r12d
    594b:	e9 37 fe ff ff       	jmpq   5787 <jerasure_bitmatrix_decode+0x2e7>
    5950:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    5955:	41 bc ff ff ff ff    	mov    $0xffffffff,%r12d
    595b:	e8 10 b9 ff ff       	callq  1270 <free@plt>
    5960:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    5965:	e8 06 b9 ff ff       	callq  1270 <free@plt>
    596a:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    596f:	e8 fc b8 ff ff       	callq  1270 <free@plt>
    5974:	e9 0e fe ff ff       	jmpq   5787 <jerasure_bitmatrix_decode+0x2e7>
    5979:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    597e:	41 83 cc ff          	or     $0xffffffff,%r12d
    5982:	e8 e9 b8 ff ff       	callq  1270 <free@plt>
    5987:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    598c:	e8 df b8 ff ff       	callq  1270 <free@plt>
    5991:	e9 f1 fd ff ff       	jmpq   5787 <jerasure_bitmatrix_decode+0x2e7>
    5996:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    599b:	41 83 cc ff          	or     $0xffffffff,%r12d
    599f:	e8 cc b8 ff ff       	callq  1270 <free@plt>
    59a4:	e9 de fd ff ff       	jmpq   5787 <jerasure_bitmatrix_decode+0x2e7>
    59a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000059b0 <jerasure_invertible_bitmatrix>:
    59b0:	f3 0f 1e fa          	endbr64 
    59b4:	41 57                	push   %r15
    59b6:	41 56                	push   %r14
    59b8:	41 55                	push   %r13
    59ba:	41 54                	push   %r12
    59bc:	55                   	push   %rbp
    59bd:	53                   	push   %rbx
    59be:	48 89 7c 24 f0       	mov    %rdi,-0x10(%rsp)
    59c3:	85 f6                	test   %esi,%esi
    59c5:	0f 8e 75 01 00 00    	jle    5b40 <jerasure_invertible_bitmatrix+0x190>
    59cb:	4c 8b 7c 24 f0       	mov    -0x10(%rsp),%r15
    59d0:	8d 5e ff             	lea    -0x1(%rsi),%ebx
    59d3:	48 63 ce             	movslq %esi,%rcx
    59d6:	45 31 f6             	xor    %r14d,%r14d
    59d9:	48 89 da             	mov    %rbx,%rdx
    59dc:	48 89 5c 24 f8       	mov    %rbx,-0x8(%rsp)
    59e1:	48 8d 04 8d 04 00 00 	lea    0x4(,%rcx,4),%rax
    59e8:	00 
    59e9:	41 83 c6 01          	add    $0x1,%r14d
    59ed:	4d 8d 6c 9f 04       	lea    0x4(%r15,%rbx,4),%r13
    59f2:	45 8b 27             	mov    (%r15),%r12d
    59f5:	48 f7 d3             	not    %rbx
    59f8:	48 89 44 24 e0       	mov    %rax,-0x20(%rsp)
    59fd:	48 8d 1c 9d 00 00 00 	lea    0x0(,%rbx,4),%rbx
    5a04:	00 
    5a05:	c7 44 24 d4 00 00 00 	movl   $0x0,-0x2c(%rsp)
    5a0c:	00 
    5a0d:	48 f7 da             	neg    %rdx
    5a10:	48 8d 04 8d 00 00 00 	lea    0x0(,%rcx,4),%rax
    5a17:	00 
    5a18:	48 c7 44 24 d8 00 00 	movq   $0x0,-0x28(%rsp)
    5a1f:	00 00 
    5a21:	48 89 5c 24 e8       	mov    %rbx,-0x18(%rsp)
    5a26:	45 85 e4             	test   %r12d,%r12d
    5a29:	0f 84 8a 00 00 00    	je     5ab9 <jerasure_invertible_bitmatrix+0x109>
    5a2f:	90                   	nop
    5a30:	44 39 f6             	cmp    %r14d,%esi
    5a33:	0f 84 07 01 00 00    	je     5b40 <jerasure_invertible_bitmatrix+0x190>
    5a39:	48 89 cb             	mov    %rcx,%rbx
    5a3c:	49 01 c5             	add    %rax,%r13
    5a3f:	01 74 24 d4          	add    %esi,-0x2c(%rsp)
    5a43:	4c 8b 64 24 e8       	mov    -0x18(%rsp),%r12
    5a48:	48 01 4c 24 d8       	add    %rcx,-0x28(%rsp)
    5a4d:	48 f7 db             	neg    %rbx
    5a50:	4d 89 eb             	mov    %r13,%r11
    5a53:	44 89 f5             	mov    %r14d,%ebp
    5a56:	eb 15                	jmp    5a6d <jerasure_invertible_bitmatrix+0xbd>
    5a58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    5a5f:	00 
    5a60:	83 c5 01             	add    $0x1,%ebp
    5a63:	48 29 cb             	sub    %rcx,%rbx
    5a66:	49 01 c3             	add    %rax,%r11
    5a69:	39 ee                	cmp    %ebp,%esi
    5a6b:	74 33                	je     5aa0 <jerasure_invertible_bitmatrix+0xf0>
    5a6d:	41 8b 7c 93 fc       	mov    -0x4(%r11,%rdx,4),%edi
    5a72:	85 ff                	test   %edi,%edi
    5a74:	74 ea                	je     5a60 <jerasure_invertible_bitmatrix+0xb0>
    5a76:	48 89 44 24 c8       	mov    %rax,-0x38(%rsp)
    5a7b:	4b 8d 3c 1c          	lea    (%r12,%r11,1),%rdi
    5a7f:	90                   	nop
    5a80:	8b 04 9f             	mov    (%rdi,%rbx,4),%eax
    5a83:	31 07                	xor    %eax,(%rdi)
    5a85:	48 83 c7 04          	add    $0x4,%rdi
    5a89:	4c 39 df             	cmp    %r11,%rdi
    5a8c:	75 f2                	jne    5a80 <jerasure_invertible_bitmatrix+0xd0>
    5a8e:	48 8b 44 24 c8       	mov    -0x38(%rsp),%rax
    5a93:	83 c5 01             	add    $0x1,%ebp
    5a96:	48 29 cb             	sub    %rcx,%rbx
    5a99:	49 01 c3             	add    %rax,%r11
    5a9c:	39 ee                	cmp    %ebp,%esi
    5a9e:	75 cd                	jne    5a6d <jerasure_invertible_bitmatrix+0xbd>
    5aa0:	4c 03 7c 24 e0       	add    -0x20(%rsp),%r15
    5aa5:	48 83 c2 01          	add    $0x1,%rdx
    5aa9:	41 83 c6 01          	add    $0x1,%r14d
    5aad:	45 8b 27             	mov    (%r15),%r12d
    5ab0:	45 85 e4             	test   %r12d,%r12d
    5ab3:	0f 85 77 ff ff ff    	jne    5a30 <jerasure_invertible_bitmatrix+0x80>
    5ab9:	44 39 f6             	cmp    %r14d,%esi
    5abc:	0f 8e 92 00 00 00    	jle    5b54 <jerasure_invertible_bitmatrix+0x1a4>
    5ac2:	8b 5c 24 d4          	mov    -0x2c(%rsp),%ebx
    5ac6:	44 89 f5             	mov    %r14d,%ebp
    5ac9:	44 8d 1c 1e          	lea    (%rsi,%rbx,1),%r11d
    5acd:	48 8b 5c 24 f8       	mov    -0x8(%rsp),%rbx
    5ad2:	48 8d 3c 1a          	lea    (%rdx,%rbx,1),%rdi
    5ad6:	49 63 db             	movslq %r11d,%rbx
    5ad9:	48 01 df             	add    %rbx,%rdi
    5adc:	48 8b 5c 24 f0       	mov    -0x10(%rsp),%rbx
    5ae1:	48 8d 1c bb          	lea    (%rbx,%rdi,4),%rbx
    5ae5:	eb 16                	jmp    5afd <jerasure_invertible_bitmatrix+0x14d>
    5ae7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    5aee:	00 00 
    5af0:	83 c5 01             	add    $0x1,%ebp
    5af3:	41 01 f3             	add    %esi,%r11d
    5af6:	48 01 c3             	add    %rax,%rbx
    5af9:	39 ee                	cmp    %ebp,%esi
    5afb:	74 49                	je     5b46 <jerasure_invertible_bitmatrix+0x196>
    5afd:	44 8b 23             	mov    (%rbx),%r12d
    5b00:	44 89 df             	mov    %r11d,%edi
    5b03:	45 85 e4             	test   %r12d,%r12d
    5b06:	74 e8                	je     5af0 <jerasure_invertible_bitmatrix+0x140>
    5b08:	48 8b 5c 24 d8       	mov    -0x28(%rsp),%rbx
    5b0d:	4c 8b 5c 24 f0       	mov    -0x10(%rsp),%r11
    5b12:	48 63 ff             	movslq %edi,%rdi
    5b15:	4d 8d 1c 9b          	lea    (%r11,%rbx,4),%r11
    5b19:	48 29 df             	sub    %rbx,%rdi
    5b1c:	0f 1f 40 00          	nopl   0x0(%rax)
    5b20:	41 8b 1b             	mov    (%r11),%ebx
    5b23:	41 8b 2c bb          	mov    (%r11,%rdi,4),%ebp
    5b27:	41 89 2b             	mov    %ebp,(%r11)
    5b2a:	41 89 1c bb          	mov    %ebx,(%r11,%rdi,4)
    5b2e:	49 83 c3 04          	add    $0x4,%r11
    5b32:	4d 39 eb             	cmp    %r13,%r11
    5b35:	75 e9                	jne    5b20 <jerasure_invertible_bitmatrix+0x170>
    5b37:	44 39 f6             	cmp    %r14d,%esi
    5b3a:	0f 85 f9 fe ff ff    	jne    5a39 <jerasure_invertible_bitmatrix+0x89>
    5b40:	41 bc 01 00 00 00    	mov    $0x1,%r12d
    5b46:	5b                   	pop    %rbx
    5b47:	44 89 e0             	mov    %r12d,%eax
    5b4a:	5d                   	pop    %rbp
    5b4b:	41 5c                	pop    %r12
    5b4d:	41 5d                	pop    %r13
    5b4f:	41 5e                	pop    %r14
    5b51:	41 5f                	pop    %r15
    5b53:	c3                   	retq   
    5b54:	74 f0                	je     5b46 <jerasure_invertible_bitmatrix+0x196>
    5b56:	89 f7                	mov    %esi,%edi
    5b58:	41 0f af fe          	imul   %r14d,%edi
    5b5c:	eb aa                	jmp    5b08 <jerasure_invertible_bitmatrix+0x158>
    5b5e:	66 90                	xchg   %ax,%ax

0000000000005b60 <jerasure_matrix_multiply>:
    5b60:	f3 0f 1e fa          	endbr64 
    5b64:	41 57                	push   %r15
    5b66:	49 63 c1             	movslq %r9d,%rax
    5b69:	41 56                	push   %r14
    5b6b:	4c 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%r15
    5b72:	00 
    5b73:	49 89 c6             	mov    %rax,%r14
    5b76:	41 55                	push   %r13
    5b78:	41 54                	push   %r12
    5b7a:	55                   	push   %rbp
    5b7b:	53                   	push   %rbx
    5b7c:	48 83 ec 58          	sub    $0x58,%rsp
    5b80:	48 89 7c 24 40       	mov    %rdi,0x40(%rsp)
    5b85:	48 63 fa             	movslq %edx,%rdi
    5b88:	44 8b a4 24 90 00 00 	mov    0x90(%rsp),%r12d
    5b8f:	00 
    5b90:	89 7c 24 3c          	mov    %edi,0x3c(%rsp)
    5b94:	48 89 fb             	mov    %rdi,%rbx
    5b97:	49 0f af ff          	imul   %r15,%rdi
    5b9b:	48 89 74 24 28       	mov    %rsi,0x28(%rsp)
    5ba0:	89 4c 24 48          	mov    %ecx,0x48(%rsp)
    5ba4:	44 89 44 24 10       	mov    %r8d,0x10(%rsp)
    5ba9:	89 44 24 34          	mov    %eax,0x34(%rsp)
    5bad:	e8 3e b8 ff ff       	callq  13f0 <malloc@plt>
    5bb2:	89 da                	mov    %ebx,%edx
    5bb4:	41 0f af d6          	imul   %r14d,%edx
    5bb8:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    5bbd:	85 d2                	test   %edx,%edx
    5bbf:	7e 26                	jle    5be7 <jerasure_matrix_multiply+0x87>
    5bc1:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
    5bc6:	83 ea 01             	sub    $0x1,%edx
    5bc9:	48 89 f0             	mov    %rsi,%rax
    5bcc:	48 8d 54 96 04       	lea    0x4(%rsi,%rdx,4),%rdx
    5bd1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    5bd8:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
    5bde:	48 83 c0 04          	add    $0x4,%rax
    5be2:	48 39 d0             	cmp    %rdx,%rax
    5be5:	75 f1                	jne    5bd8 <jerasure_matrix_multiply+0x78>
    5be7:	8b 4c 24 3c          	mov    0x3c(%rsp),%ecx
    5beb:	85 c9                	test   %ecx,%ecx
    5bed:	0f 8e f8 00 00 00    	jle    5ceb <jerasure_matrix_multiply+0x18b>
    5bf3:	8b 44 24 34          	mov    0x34(%rsp),%eax
    5bf7:	c7 44 24 14 00 00 00 	movl   $0x0,0x14(%rsp)
    5bfe:	00 
    5bff:	c7 44 24 38 00 00 00 	movl   $0x0,0x38(%rsp)
    5c06:	00 
    5c07:	83 e8 01             	sub    $0x1,%eax
    5c0a:	c7 44 24 30 00 00 00 	movl   $0x0,0x30(%rsp)
    5c11:	00 
    5c12:	89 44 24 4c          	mov    %eax,0x4c(%rsp)
    5c16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    5c1d:	00 00 00 
    5c20:	8b 44 24 34          	mov    0x34(%rsp),%eax
    5c24:	85 c0                	test   %eax,%eax
    5c26:	0f 8e 9c 00 00 00    	jle    5cc8 <jerasure_matrix_multiply+0x168>
    5c2c:	8b 44 24 4c          	mov    0x4c(%rsp),%eax
    5c30:	48 63 54 24 38       	movslq 0x38(%rsp),%rdx
    5c35:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
    5c3c:	00 
    5c3d:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    5c42:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    5c47:	48 8d 04 97          	lea    (%rdi,%rdx,4),%rax
    5c4b:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    5c50:	8b 44 24 10          	mov    0x10(%rsp),%eax
    5c54:	83 e8 01             	sub    $0x1,%eax
    5c57:	48 01 d0             	add    %rdx,%rax
    5c5a:	48 8d 6c 87 04       	lea    0x4(%rdi,%rax,4),%rbp
    5c5f:	90                   	nop
    5c60:	8b 54 24 10          	mov    0x10(%rsp),%edx
    5c64:	85 d2                	test   %edx,%edx
    5c66:	7e 45                	jle    5cad <jerasure_matrix_multiply+0x14d>
    5c68:	48 8b 0c 24          	mov    (%rsp),%rcx
    5c6c:	8b 44 24 14          	mov    0x14(%rsp),%eax
    5c70:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    5c75:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
    5c7a:	01 c8                	add    %ecx,%eax
    5c7c:	48 98                	cltq   
    5c7e:	4c 8d 2c 87          	lea    (%rdi,%rax,4),%r13
    5c82:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    5c87:	4c 8d 34 88          	lea    (%rax,%rcx,4),%r14
    5c8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5c90:	41 8b 36             	mov    (%r14),%esi
    5c93:	8b 3b                	mov    (%rbx),%edi
    5c95:	44 89 e2             	mov    %r12d,%edx
    5c98:	48 83 c3 04          	add    $0x4,%rbx
    5c9c:	4d 01 fe             	add    %r15,%r14
    5c9f:	e8 6c 28 00 00       	callq  8510 <galois_single_multiply>
    5ca4:	41 31 45 00          	xor    %eax,0x0(%r13)
    5ca8:	48 39 eb             	cmp    %rbp,%rbx
    5cab:	75 e3                	jne    5c90 <jerasure_matrix_multiply+0x130>
    5cad:	48 8b 0c 24          	mov    (%rsp),%rcx
    5cb1:	48 8d 41 01          	lea    0x1(%rcx),%rax
    5cb5:	48 39 4c 24 08       	cmp    %rcx,0x8(%rsp)
    5cba:	74 0c                	je     5cc8 <jerasure_matrix_multiply+0x168>
    5cbc:	48 89 04 24          	mov    %rax,(%rsp)
    5cc0:	eb 9e                	jmp    5c60 <jerasure_matrix_multiply+0x100>
    5cc2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    5cc8:	8b 74 24 48          	mov    0x48(%rsp),%esi
    5ccc:	83 44 24 30 01       	addl   $0x1,0x30(%rsp)
    5cd1:	01 74 24 38          	add    %esi,0x38(%rsp)
    5cd5:	8b 44 24 30          	mov    0x30(%rsp),%eax
    5cd9:	8b 74 24 34          	mov    0x34(%rsp),%esi
    5cdd:	01 74 24 14          	add    %esi,0x14(%rsp)
    5ce1:	39 44 24 3c          	cmp    %eax,0x3c(%rsp)
    5ce5:	0f 85 35 ff ff ff    	jne    5c20 <jerasure_matrix_multiply+0xc0>
    5ceb:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    5cf0:	48 83 c4 58          	add    $0x58,%rsp
    5cf4:	5b                   	pop    %rbx
    5cf5:	5d                   	pop    %rbp
    5cf6:	41 5c                	pop    %r12
    5cf8:	41 5d                	pop    %r13
    5cfa:	41 5e                	pop    %r14
    5cfc:	41 5f                	pop    %r15
    5cfe:	c3                   	retq   
    5cff:	90                   	nop

0000000000005d00 <jerasure_get_stats>:
    5d00:	f3 0f 1e fa          	endbr64 
    5d04:	f2 0f 10 05 54 b4 00 	movsd  0xb454(%rip),%xmm0        # 11160 <jerasure_total_xor_bytes>
    5d0b:	00 
    5d0c:	48 c7 05 49 b4 00 00 	movq   $0x0,0xb449(%rip)        # 11160 <jerasure_total_xor_bytes>
    5d13:	00 00 00 00 
    5d17:	f2 0f 11 07          	movsd  %xmm0,(%rdi)
    5d1b:	f2 0f 10 05 35 b4 00 	movsd  0xb435(%rip),%xmm0        # 11158 <jerasure_total_gf_bytes>
    5d22:	00 
    5d23:	48 c7 05 2a b4 00 00 	movq   $0x0,0xb42a(%rip)        # 11158 <jerasure_total_gf_bytes>
    5d2a:	00 00 00 00 
    5d2e:	f2 0f 11 47 08       	movsd  %xmm0,0x8(%rdi)
    5d33:	f2 0f 10 05 15 b4 00 	movsd  0xb415(%rip),%xmm0        # 11150 <jerasure_total_memcpy_bytes>
    5d3a:	00 
    5d3b:	48 c7 05 0a b4 00 00 	movq   $0x0,0xb40a(%rip)        # 11150 <jerasure_total_memcpy_bytes>
    5d42:	00 00 00 00 
    5d46:	f2 0f 11 47 10       	movsd  %xmm0,0x10(%rdi)
    5d4b:	c3                   	retq   
    5d4c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000005d50 <jerasure_do_scheduled_operations>:
    5d50:	f3 0f 1e fa          	endbr64 
    5d54:	41 57                	push   %r15
    5d56:	41 56                	push   %r14
    5d58:	41 55                	push   %r13
    5d5a:	41 54                	push   %r12
    5d5c:	55                   	push   %rbp
    5d5d:	53                   	push   %rbx
    5d5e:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
    5d63:	85 d2                	test   %edx,%edx
    5d65:	0f 8e 20 01 00 00    	jle    5e8b <jerasure_do_scheduled_operations+0x13b>
    5d6b:	48 8b 36             	mov    (%rsi),%rsi
    5d6e:	89 d3                	mov    %edx,%ebx
    5d70:	48 89 74 24 f8       	mov    %rsi,-0x8(%rsp)
    5d75:	8b 16                	mov    (%rsi),%edx
    5d77:	89 54 24 ec          	mov    %edx,-0x14(%rsp)
    5d7b:	31 d2                	xor    %edx,%edx
    5d7d:	0f 1f 00             	nopl   (%rax)
    5d80:	8b 4c 24 ec          	mov    -0x14(%rsp),%ecx
    5d84:	85 c9                	test   %ecx,%ecx
    5d86:	0f 88 14 01 00 00    	js     5ea0 <jerasure_do_scheduled_operations+0x150>
    5d8c:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
    5d91:	48 63 6c 24 ec       	movslq -0x14(%rsp),%rbp
    5d96:	31 c9                	xor    %ecx,%ecx
    5d98:	4c 8b 5c 24 f8       	mov    -0x8(%rsp),%r11
    5d9d:	48 8d 70 08          	lea    0x8(%rax),%rsi
    5da1:	eb 36                	jmp    5dd9 <jerasure_do_scheduled_operations+0x89>
    5da3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5da8:	4c 33 00             	xor    (%rax),%r8
    5dab:	4c 33 48 08          	xor    0x8(%rax),%r9
    5daf:	4c 33 50 10          	xor    0x10(%rax),%r10
    5db3:	4c 33 58 18          	xor    0x18(%rax),%r11
    5db7:	4c 33 60 20          	xor    0x20(%rax),%r12
    5dbb:	4c 33 68 28          	xor    0x28(%rax),%r13
    5dbf:	4c 33 70 30          	xor    0x30(%rax),%r14
    5dc3:	4c 33 78 38          	xor    0x38(%rax),%r15
    5dc7:	4c 8b 1e             	mov    (%rsi),%r11
    5dca:	48 83 c6 08          	add    $0x8,%rsi
    5dce:	49 63 2b             	movslq (%r11),%rbp
    5dd1:	85 ed                	test   %ebp,%ebp
    5dd3:	0f 88 87 00 00 00    	js     5e60 <jerasure_do_scheduled_operations+0x110>
    5dd9:	41 8b 43 04          	mov    0x4(%r11),%eax
    5ddd:	4d 63 63 08          	movslq 0x8(%r11),%r12
    5de1:	0f af c3             	imul   %ebx,%eax
    5de4:	48 98                	cltq   
    5de6:	48 01 d0             	add    %rdx,%rax
    5de9:	48 03 04 ef          	add    (%rdi,%rbp,8),%rax
    5ded:	48 89 cd             	mov    %rcx,%rbp
    5df0:	41 8b 4b 0c          	mov    0xc(%r11),%ecx
    5df4:	45 8b 5b 10          	mov    0x10(%r11),%r11d
    5df8:	0f af cb             	imul   %ebx,%ecx
    5dfb:	48 63 c9             	movslq %ecx,%rcx
    5dfe:	48 01 d1             	add    %rdx,%rcx
    5e01:	4a 03 0c e7          	add    (%rdi,%r12,8),%rcx
    5e05:	45 85 db             	test   %r11d,%r11d
    5e08:	75 9e                	jne    5da8 <jerasure_do_scheduled_operations+0x58>
    5e0a:	48 85 ed             	test   %rbp,%rbp
    5e0d:	74 20                	je     5e2f <jerasure_do_scheduled_operations+0xdf>
    5e0f:	4c 89 45 00          	mov    %r8,0x0(%rbp)
    5e13:	4c 89 4d 08          	mov    %r9,0x8(%rbp)
    5e17:	4c 89 55 10          	mov    %r10,0x10(%rbp)
    5e1b:	4c 89 5d 18          	mov    %r11,0x18(%rbp)
    5e1f:	4c 89 65 20          	mov    %r12,0x20(%rbp)
    5e23:	4c 89 6d 28          	mov    %r13,0x28(%rbp)
    5e27:	4c 89 75 30          	mov    %r14,0x30(%rbp)
    5e2b:	4c 89 7d 38          	mov    %r15,0x38(%rbp)
    5e2f:	4c 8b 00             	mov    (%rax),%r8
    5e32:	4c 8b 48 08          	mov    0x8(%rax),%r9
    5e36:	4c 8b 50 10          	mov    0x10(%rax),%r10
    5e3a:	4c 8b 58 18          	mov    0x18(%rax),%r11
    5e3e:	4c 8b 60 20          	mov    0x20(%rax),%r12
    5e42:	4c 8b 68 28          	mov    0x28(%rax),%r13
    5e46:	4c 8b 70 30          	mov    0x30(%rax),%r14
    5e4a:	4c 8b 78 38          	mov    0x38(%rax),%r15
    5e4e:	4c 8b 1e             	mov    (%rsi),%r11
    5e51:	48 83 c6 08          	add    $0x8,%rsi
    5e55:	49 63 2b             	movslq (%r11),%rbp
    5e58:	85 ed                	test   %ebp,%ebp
    5e5a:	0f 89 79 ff ff ff    	jns    5dd9 <jerasure_do_scheduled_operations+0x89>
    5e60:	4c 8b 00             	mov    (%rax),%r8
    5e63:	4c 8b 48 08          	mov    0x8(%rax),%r9
    5e67:	4c 8b 50 10          	mov    0x10(%rax),%r10
    5e6b:	4c 8b 58 18          	mov    0x18(%rax),%r11
    5e6f:	4c 8b 60 20          	mov    0x20(%rax),%r12
    5e73:	4c 8b 68 28          	mov    0x28(%rax),%r13
    5e77:	4c 8b 70 30          	mov    0x30(%rax),%r14
    5e7b:	4c 8b 78 38          	mov    0x38(%rax),%r15
    5e7f:	48 83 c2 40          	add    $0x40,%rdx
    5e83:	39 d3                	cmp    %edx,%ebx
    5e85:	0f 8f f5 fe ff ff    	jg     5d80 <jerasure_do_scheduled_operations+0x30>
    5e8b:	5b                   	pop    %rbx
    5e8c:	5d                   	pop    %rbp
    5e8d:	41 5c                	pop    %r12
    5e8f:	41 5d                	pop    %r13
    5e91:	41 5e                	pop    %r14
    5e93:	41 5f                	pop    %r15
    5e95:	c3                   	retq   
    5e96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    5e9d:	00 00 00 
    5ea0:	31 c9                	xor    %ecx,%ecx
    5ea2:	eb bc                	jmp    5e60 <jerasure_do_scheduled_operations+0x110>
    5ea4:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    5eab:	00 00 00 00 
    5eaf:	90                   	nop

0000000000005eb0 <jerasure_schedule_decode_cache>:
    5eb0:	f3 0f 1e fa          	endbr64 
    5eb4:	41 57                	push   %r15
    5eb6:	4d 89 cb             	mov    %r9,%r11
    5eb9:	41 56                	push   %r14
    5ebb:	41 55                	push   %r13
    5ebd:	41 54                	push   %r12
    5ebf:	55                   	push   %rbp
    5ec0:	53                   	push   %rbx
    5ec1:	89 d3                	mov    %edx,%ebx
    5ec3:	4c 89 c2             	mov    %r8,%rdx
    5ec6:	48 83 ec 18          	sub    $0x18,%rsp
    5eca:	45 8b 70 04          	mov    0x4(%r8),%r14d
    5ece:	44 8b 6c 24 60       	mov    0x60(%rsp),%r13d
    5ed3:	41 83 fe ff          	cmp    $0xffffffff,%r14d
    5ed7:	0f 84 99 00 00 00    	je     5f76 <jerasure_schedule_decode_cache+0xc6>
    5edd:	41 83 78 08 ff       	cmpl   $0xffffffff,0x8(%r8)
    5ee2:	0f 85 a2 00 00 00    	jne    5f8a <jerasure_schedule_decode_cache+0xda>
    5ee8:	41 8b 00             	mov    (%r8),%eax
    5eeb:	8d 2c 37             	lea    (%rdi,%rsi,1),%ebp
    5eee:	0f af c5             	imul   %ebp,%eax
    5ef1:	44 01 f0             	add    %r14d,%eax
    5ef4:	48 98                	cltq   
    5ef6:	4c 8b 44 24 50       	mov    0x50(%rsp),%r8
    5efb:	4c 8b 34 c1          	mov    (%rcx,%rax,8),%r14
    5eff:	4c 89 d9             	mov    %r11,%rcx
    5f02:	e8 39 e6 ff ff       	callq  4540 <set_up_ptrs_for_scheduled_decoding>
    5f07:	48 89 c7             	mov    %rax,%rdi
    5f0a:	48 85 c0             	test   %rax,%rax
    5f0d:	74 7b                	je     5f8a <jerasure_schedule_decode_cache+0xda>
    5f0f:	8b 44 24 58          	mov    0x58(%rsp),%eax
    5f13:	85 c0                	test   %eax,%eax
    5f15:	7e 49                	jle    5f60 <jerasure_schedule_decode_cache+0xb0>
    5f17:	41 0f af dd          	imul   %r13d,%ebx
    5f1b:	8d 45 ff             	lea    -0x1(%rbp),%eax
    5f1e:	45 31 e4             	xor    %r12d,%r12d
    5f21:	4c 8d 7c c7 08       	lea    0x8(%rdi,%rax,8),%r15
    5f26:	89 5c 24 0c          	mov    %ebx,0xc(%rsp)
    5f2a:	48 63 db             	movslq %ebx,%rbx
    5f2d:	0f 1f 00             	nopl   (%rax)
    5f30:	44 89 ea             	mov    %r13d,%edx
    5f33:	4c 89 f6             	mov    %r14,%rsi
    5f36:	e8 15 fe ff ff       	callq  5d50 <jerasure_do_scheduled_operations>
    5f3b:	48 89 fa             	mov    %rdi,%rdx
    5f3e:	85 ed                	test   %ebp,%ebp
    5f40:	7e 12                	jle    5f54 <jerasure_schedule_decode_cache+0xa4>
    5f42:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    5f48:	48 01 1a             	add    %rbx,(%rdx)
    5f4b:	48 83 c2 08          	add    $0x8,%rdx
    5f4f:	49 39 d7             	cmp    %rdx,%r15
    5f52:	75 f4                	jne    5f48 <jerasure_schedule_decode_cache+0x98>
    5f54:	44 03 64 24 0c       	add    0xc(%rsp),%r12d
    5f59:	44 39 64 24 58       	cmp    %r12d,0x58(%rsp)
    5f5e:	7f d0                	jg     5f30 <jerasure_schedule_decode_cache+0x80>
    5f60:	e8 0b b3 ff ff       	callq  1270 <free@plt>
    5f65:	31 c0                	xor    %eax,%eax
    5f67:	48 83 c4 18          	add    $0x18,%rsp
    5f6b:	5b                   	pop    %rbx
    5f6c:	5d                   	pop    %rbp
    5f6d:	41 5c                	pop    %r12
    5f6f:	41 5d                	pop    %r13
    5f71:	41 5e                	pop    %r14
    5f73:	41 5f                	pop    %r15
    5f75:	c3                   	retq   
    5f76:	45 8b 30             	mov    (%r8),%r14d
    5f79:	8d 2c 37             	lea    (%rdi,%rsi,1),%ebp
    5f7c:	89 e8                	mov    %ebp,%eax
    5f7e:	41 0f af c6          	imul   %r14d,%eax
    5f82:	44 01 f0             	add    %r14d,%eax
    5f85:	e9 6a ff ff ff       	jmpq   5ef4 <jerasure_schedule_decode_cache+0x44>
    5f8a:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    5f8f:	eb d6                	jmp    5f67 <jerasure_schedule_decode_cache+0xb7>
    5f91:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    5f98:	00 00 00 00 
    5f9c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000005fa0 <jerasure_schedule_encode>:
    5fa0:	f3 0f 1e fa          	endbr64 
    5fa4:	41 57                	push   %r15
    5fa6:	41 89 f7             	mov    %esi,%r15d
    5fa9:	41 56                	push   %r14
    5fab:	4c 63 f7             	movslq %edi,%r14
    5fae:	41 55                	push   %r13
    5fb0:	41 54                	push   %r12
    5fb2:	4d 89 cc             	mov    %r9,%r12
    5fb5:	55                   	push   %rbp
    5fb6:	41 8d 2c 36          	lea    (%r14,%rsi,1),%ebp
    5fba:	53                   	push   %rbx
    5fbb:	48 63 fd             	movslq %ebp,%rdi
    5fbe:	48 89 cb             	mov    %rcx,%rbx
    5fc1:	48 c1 e7 03          	shl    $0x3,%rdi
    5fc5:	48 83 ec 18          	sub    $0x18,%rsp
    5fc9:	8b 44 24 50          	mov    0x50(%rsp),%eax
    5fcd:	44 8b 6c 24 58       	mov    0x58(%rsp),%r13d
    5fd2:	89 54 24 0c          	mov    %edx,0xc(%rsp)
    5fd6:	4c 89 04 24          	mov    %r8,(%rsp)
    5fda:	89 44 24 08          	mov    %eax,0x8(%rsp)
    5fde:	e8 0d b4 ff ff       	callq  13f0 <malloc@plt>
    5fe3:	31 d2                	xor    %edx,%edx
    5fe5:	45 85 f6             	test   %r14d,%r14d
    5fe8:	48 8b 34 24          	mov    (%rsp),%rsi
    5fec:	48 89 c7             	mov    %rax,%rdi
    5fef:	45 8d 5e ff          	lea    -0x1(%r14),%r11d
    5ff3:	7e 17                	jle    600c <jerasure_schedule_encode+0x6c>
    5ff5:	0f 1f 00             	nopl   (%rax)
    5ff8:	48 8b 04 d6          	mov    (%rsi,%rdx,8),%rax
    5ffc:	48 89 04 d7          	mov    %rax,(%rdi,%rdx,8)
    6000:	48 89 d0             	mov    %rdx,%rax
    6003:	48 83 c2 01          	add    $0x1,%rdx
    6007:	49 39 c3             	cmp    %rax,%r11
    600a:	75 ec                	jne    5ff8 <jerasure_schedule_encode+0x58>
    600c:	45 85 ff             	test   %r15d,%r15d
    600f:	7e 23                	jle    6034 <jerasure_schedule_encode+0x94>
    6011:	41 8d 77 ff          	lea    -0x1(%r15),%esi
    6015:	4e 8d 1c f7          	lea    (%rdi,%r14,8),%r11
    6019:	31 c0                	xor    %eax,%eax
    601b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    6020:	49 8b 14 c4          	mov    (%r12,%rax,8),%rdx
    6024:	49 89 14 c3          	mov    %rdx,(%r11,%rax,8)
    6028:	48 89 c2             	mov    %rax,%rdx
    602b:	48 83 c0 01          	add    $0x1,%rax
    602f:	48 39 d6             	cmp    %rdx,%rsi
    6032:	75 ec                	jne    6020 <jerasure_schedule_encode+0x80>
    6034:	8b 44 24 08          	mov    0x8(%rsp),%eax
    6038:	85 c0                	test   %eax,%eax
    603a:	7e 61                	jle    609d <jerasure_schedule_encode+0xfd>
    603c:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    6040:	8d 55 ff             	lea    -0x1(%rbp),%edx
    6043:	45 31 f6             	xor    %r14d,%r14d
    6046:	4c 8d 7c d7 08       	lea    0x8(%rdi,%rdx,8),%r15
    604b:	41 0f af c5          	imul   %r13d,%eax
    604f:	4c 63 e0             	movslq %eax,%r12
    6052:	89 04 24             	mov    %eax,(%rsp)
    6055:	44 89 e8             	mov    %r13d,%eax
    6058:	4d 89 fd             	mov    %r15,%r13
    605b:	4d 89 e7             	mov    %r12,%r15
    605e:	49 89 dc             	mov    %rbx,%r12
    6061:	44 89 f3             	mov    %r14d,%ebx
    6064:	41 89 c6             	mov    %eax,%r14d
    6067:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    606e:	00 00 
    6070:	44 89 f2             	mov    %r14d,%edx
    6073:	4c 89 e6             	mov    %r12,%rsi
    6076:	e8 d5 fc ff ff       	callq  5d50 <jerasure_do_scheduled_operations>
    607b:	48 89 f8             	mov    %rdi,%rax
    607e:	85 ed                	test   %ebp,%ebp
    6080:	7e 12                	jle    6094 <jerasure_schedule_encode+0xf4>
    6082:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    6088:	4c 01 38             	add    %r15,(%rax)
    608b:	48 83 c0 08          	add    $0x8,%rax
    608f:	49 39 c5             	cmp    %rax,%r13
    6092:	75 f4                	jne    6088 <jerasure_schedule_encode+0xe8>
    6094:	03 1c 24             	add    (%rsp),%ebx
    6097:	39 5c 24 08          	cmp    %ebx,0x8(%rsp)
    609b:	7f d3                	jg     6070 <jerasure_schedule_encode+0xd0>
    609d:	48 83 c4 18          	add    $0x18,%rsp
    60a1:	5b                   	pop    %rbx
    60a2:	5d                   	pop    %rbp
    60a3:	41 5c                	pop    %r12
    60a5:	41 5d                	pop    %r13
    60a7:	41 5e                	pop    %r14
    60a9:	41 5f                	pop    %r15
    60ab:	e9 c0 b1 ff ff       	jmpq   1270 <free@plt>

00000000000060b0 <jerasure_dumb_bitmatrix_to_schedule>:
    60b0:	f3 0f 1e fa          	endbr64 
    60b4:	41 57                	push   %r15
    60b6:	41 56                	push   %r14
    60b8:	41 89 fe             	mov    %edi,%r14d
    60bb:	41 55                	push   %r13
    60bd:	41 54                	push   %r12
    60bf:	55                   	push   %rbp
    60c0:	89 d5                	mov    %edx,%ebp
    60c2:	53                   	push   %rbx
    60c3:	89 f3                	mov    %esi,%ebx
    60c5:	0f af dd             	imul   %ebp,%ebx
    60c8:	48 83 ec 48          	sub    $0x48,%rsp
    60cc:	89 7c 24 20          	mov    %edi,0x20(%rsp)
    60d0:	0f af fe             	imul   %esi,%edi
    60d3:	89 54 24 24          	mov    %edx,0x24(%rsp)
    60d7:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
    60dc:	0f af fa             	imul   %edx,%edi
    60df:	0f af fa             	imul   %edx,%edi
    60e2:	83 c7 01             	add    $0x1,%edi
    60e5:	48 63 ff             	movslq %edi,%rdi
    60e8:	48 c1 e7 03          	shl    $0x3,%rdi
    60ec:	e8 ff b2 ff ff       	callq  13f0 <malloc@plt>
    60f1:	89 5c 24 38          	mov    %ebx,0x38(%rsp)
    60f5:	49 89 c7             	mov    %rax,%r15
    60f8:	85 db                	test   %ebx,%ebx
    60fa:	0f 8e 39 01 00 00    	jle    6239 <jerasure_dumb_bitmatrix_to_schedule+0x189>
    6100:	44 0f af f5          	imul   %ebp,%r14d
    6104:	c7 44 24 10 00 00 00 	movl   $0x0,0x10(%rsp)
    610b:	00 
    610c:	c7 44 24 2c 00 00 00 	movl   $0x0,0x2c(%rsp)
    6113:	00 
    6114:	44 89 f0             	mov    %r14d,%eax
    6117:	44 89 74 24 28       	mov    %r14d,0x28(%rsp)
    611c:	45 31 f6             	xor    %r14d,%r14d
    611f:	83 e8 01             	sub    $0x1,%eax
    6122:	89 44 24 3c          	mov    %eax,0x3c(%rsp)
    6126:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    612d:	00 00 00 
    6130:	8b 44 24 28          	mov    0x28(%rsp),%eax
    6134:	85 c0                	test   %eax,%eax
    6136:	0f 8e f4 00 00 00    	jle    6230 <jerasure_dumb_bitmatrix_to_schedule+0x180>
    613c:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    6141:	44 8b 5c 24 3c       	mov    0x3c(%rsp),%r11d
    6146:	31 db                	xor    %ebx,%ebx
    6148:	31 d2                	xor    %edx,%edx
    614a:	48 63 44 24 2c       	movslq 0x2c(%rsp),%rax
    614f:	48 8d 34 81          	lea    (%rcx,%rax,4),%rsi
    6153:	4c 89 f9             	mov    %r15,%rcx
    6156:	4d 89 df             	mov    %r11,%r15
    6159:	eb 11                	jmp    616c <jerasure_dumb_bitmatrix_to_schedule+0xbc>
    615b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    6160:	48 8d 43 01          	lea    0x1(%rbx),%rax
    6164:	49 39 df             	cmp    %rbx,%r15
    6167:	74 7b                	je     61e4 <jerasure_dumb_bitmatrix_to_schedule+0x134>
    6169:	48 89 c3             	mov    %rax,%rbx
    616c:	49 63 ee             	movslq %r14d,%rbp
    616f:	8b 3c 9e             	mov    (%rsi,%rbx,4),%edi
    6172:	48 c1 e5 03          	shl    $0x3,%rbp
    6176:	4c 8d 2c 29          	lea    (%rcx,%rbp,1),%r13
    617a:	85 ff                	test   %edi,%edi
    617c:	74 e2                	je     6160 <jerasure_dumb_bitmatrix_to_schedule+0xb0>
    617e:	bf 14 00 00 00       	mov    $0x14,%edi
    6183:	48 89 4c 24 18       	mov    %rcx,0x18(%rsp)
    6188:	41 83 c6 01          	add    $0x1,%r14d
    618c:	89 54 24 14          	mov    %edx,0x14(%rsp)
    6190:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
    6195:	e8 56 b2 ff ff       	callq  13f0 <malloc@plt>
    619a:	8b 54 24 14          	mov    0x14(%rsp),%edx
    619e:	8b 74 24 24          	mov    0x24(%rsp),%esi
    61a2:	48 89 c7             	mov    %rax,%rdi
    61a5:	49 89 45 00          	mov    %rax,0x0(%r13)
    61a9:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
    61ae:	89 50 10             	mov    %edx,0x10(%rax)
    61b1:	89 d8                	mov    %ebx,%eax
    61b3:	99                   	cltd   
    61b4:	4c 8d 6c 29 08       	lea    0x8(%rcx,%rbp,1),%r13
    61b9:	f7 fe                	idiv   %esi
    61bb:	89 07                	mov    %eax,(%rdi)
    61bd:	8b 44 24 10          	mov    0x10(%rsp),%eax
    61c1:	89 57 04             	mov    %edx,0x4(%rdi)
    61c4:	99                   	cltd   
    61c5:	f7 fe                	idiv   %esi
    61c7:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
    61cc:	03 44 24 20          	add    0x20(%rsp),%eax
    61d0:	89 47 08             	mov    %eax,0x8(%rdi)
    61d3:	48 8d 43 01          	lea    0x1(%rbx),%rax
    61d7:	89 57 0c             	mov    %edx,0xc(%rdi)
    61da:	ba 01 00 00 00       	mov    $0x1,%edx
    61df:	49 39 df             	cmp    %rbx,%r15
    61e2:	75 85                	jne    6169 <jerasure_dumb_bitmatrix_to_schedule+0xb9>
    61e4:	8b 54 24 28          	mov    0x28(%rsp),%edx
    61e8:	01 54 24 2c          	add    %edx,0x2c(%rsp)
    61ec:	49 89 cf             	mov    %rcx,%r15
    61ef:	83 44 24 10 01       	addl   $0x1,0x10(%rsp)
    61f4:	8b 44 24 10          	mov    0x10(%rsp),%eax
    61f8:	3b 44 24 38          	cmp    0x38(%rsp),%eax
    61fc:	0f 85 2e ff ff ff    	jne    6130 <jerasure_dumb_bitmatrix_to_schedule+0x80>
    6202:	bf 14 00 00 00       	mov    $0x14,%edi
    6207:	e8 e4 b1 ff ff       	callq  13f0 <malloc@plt>
    620c:	49 89 45 00          	mov    %rax,0x0(%r13)
    6210:	c7 00 ff ff ff ff    	movl   $0xffffffff,(%rax)
    6216:	48 83 c4 48          	add    $0x48,%rsp
    621a:	4c 89 f8             	mov    %r15,%rax
    621d:	5b                   	pop    %rbx
    621e:	5d                   	pop    %rbp
    621f:	41 5c                	pop    %r12
    6221:	41 5d                	pop    %r13
    6223:	41 5e                	pop    %r14
    6225:	41 5f                	pop    %r15
    6227:	c3                   	retq   
    6228:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    622f:	00 
    6230:	49 63 c6             	movslq %r14d,%rax
    6233:	4d 8d 2c c7          	lea    (%r15,%rax,8),%r13
    6237:	eb b6                	jmp    61ef <jerasure_dumb_bitmatrix_to_schedule+0x13f>
    6239:	49 89 c5             	mov    %rax,%r13
    623c:	eb c4                	jmp    6202 <jerasure_dumb_bitmatrix_to_schedule+0x152>
    623e:	66 90                	xchg   %ax,%ax

0000000000006240 <jerasure_smart_bitmatrix_to_schedule>:
    6240:	f3 0f 1e fa          	endbr64 
    6244:	41 57                	push   %r15
    6246:	41 56                	push   %r14
    6248:	41 89 d6             	mov    %edx,%r14d
    624b:	41 55                	push   %r13
    624d:	41 54                	push   %r12
    624f:	55                   	push   %rbp
    6250:	89 fd                	mov    %edi,%ebp
    6252:	53                   	push   %rbx
    6253:	89 f3                	mov    %esi,%ebx
    6255:	41 0f af ee          	imul   %r14d,%ebp
    6259:	41 0f af de          	imul   %r14d,%ebx
    625d:	48 83 ec 78          	sub    $0x78,%rsp
    6261:	89 7c 24 14          	mov    %edi,0x14(%rsp)
    6265:	0f af fe             	imul   %esi,%edi
    6268:	4c 63 eb             	movslq %ebx,%r13
    626b:	49 c1 e5 02          	shl    $0x2,%r13
    626f:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    6274:	0f af fa             	imul   %edx,%edi
    6277:	0f af fa             	imul   %edx,%edi
    627a:	83 c7 01             	add    $0x1,%edi
    627d:	48 63 ff             	movslq %edi,%rdi
    6280:	48 c1 e7 03          	shl    $0x3,%rdi
    6284:	e8 67 b1 ff ff       	callq  13f0 <malloc@plt>
    6289:	4c 89 ef             	mov    %r13,%rdi
    628c:	49 89 c7             	mov    %rax,%r15
    628f:	e8 5c b1 ff ff       	callq  13f0 <malloc@plt>
    6294:	4c 89 ef             	mov    %r13,%rdi
    6297:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    629c:	e8 4f b1 ff ff       	callq  13f0 <malloc@plt>
    62a1:	4c 89 ef             	mov    %r13,%rdi
    62a4:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    62a9:	e8 42 b1 ff ff       	callq  13f0 <malloc@plt>
    62ae:	4c 89 ef             	mov    %r13,%rdi
    62b1:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    62b6:	e8 35 b1 ff ff       	callq  13f0 <malloc@plt>
    62bb:	8d 75 01             	lea    0x1(%rbp),%esi
    62be:	89 6c 24 18          	mov    %ebp,0x18(%rsp)
    62c2:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    62c7:	89 74 24 6c          	mov    %esi,0x6c(%rsp)
    62cb:	85 db                	test   %ebx,%ebx
    62cd:	0f 8e a7 00 00 00    	jle    637a <jerasure_smart_bitmatrix_to_schedule+0x13a>
    62d3:	48 89 c7             	mov    %rax,%rdi
    62d6:	8d 45 ff             	lea    -0x1(%rbp),%eax
    62d9:	4c 89 6c 24 08       	mov    %r13,0x8(%rsp)
    62de:	4c 8b 6c 24 28       	mov    0x28(%rsp),%r13
    62e3:	44 89 74 24 38       	mov    %r14d,0x38(%rsp)
    62e8:	48 8d 2c 85 04 00 00 	lea    0x4(,%rax,4),%rbp
    62ef:	00 
    62f0:	4c 8b 74 24 40       	mov    0x40(%rsp),%r14
    62f5:	44 8d 63 ff          	lea    -0x1(%rbx),%r12d
    62f9:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    62fe:	4c 89 7c 24 30       	mov    %r15,0x30(%rsp)
    6303:	89 f3                	mov    %esi,%ebx
    6305:	49 89 ff             	mov    %rdi,%r15
    6308:	31 f6                	xor    %esi,%esi
    630a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    6310:	31 d2                	xor    %edx,%edx
    6312:	83 7c 24 18 00       	cmpl   $0x0,0x18(%rsp)
    6317:	89 f7                	mov    %esi,%edi
    6319:	41 89 f3             	mov    %esi,%r11d
    631c:	48 8d 0c 28          	lea    (%rax,%rbp,1),%rcx
    6320:	7e 11                	jle    6333 <jerasure_smart_bitmatrix_to_schedule+0xf3>
    6322:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    6328:	03 10                	add    (%rax),%edx
    632a:	48 83 c0 04          	add    $0x4,%rax
    632e:	48 39 c8             	cmp    %rcx,%rax
    6331:	75 f5                	jne    6328 <jerasure_smart_bitmatrix_to_schedule+0xe8>
    6333:	41 c7 44 b5 00 ff ff 	movl   $0xffffffff,0x0(%r13,%rsi,4)
    633a:	ff ff 
    633c:	48 8b 4c 24 50       	mov    0x50(%rsp),%rcx
    6341:	89 14 b1             	mov    %edx,(%rcx,%rsi,4)
    6344:	8d 4f 01             	lea    0x1(%rdi),%ecx
    6347:	83 ef 01             	sub    $0x1,%edi
    634a:	41 89 0c b6          	mov    %ecx,(%r14,%rsi,4)
    634e:	41 89 3c b7          	mov    %edi,(%r15,%rsi,4)
    6352:	39 d3                	cmp    %edx,%ebx
    6354:	7e 07                	jle    635d <jerasure_smart_bitmatrix_to_schedule+0x11d>
    6356:	44 89 5c 24 1c       	mov    %r11d,0x1c(%rsp)
    635b:	89 d3                	mov    %edx,%ebx
    635d:	48 8d 56 01          	lea    0x1(%rsi),%rdx
    6361:	49 39 f4             	cmp    %rsi,%r12
    6364:	74 05                	je     636b <jerasure_smart_bitmatrix_to_schedule+0x12b>
    6366:	48 89 d6             	mov    %rdx,%rsi
    6369:	eb a5                	jmp    6310 <jerasure_smart_bitmatrix_to_schedule+0xd0>
    636b:	4c 8b 6c 24 08       	mov    0x8(%rsp),%r13
    6370:	4c 8b 7c 24 30       	mov    0x30(%rsp),%r15
    6375:	44 8b 74 24 38       	mov    0x38(%rsp),%r14d
    637a:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    637f:	c7 44 24 68 00 00 00 	movl   $0x0,0x68(%rsp)
    6386:	00 
    6387:	31 f6                	xor    %esi,%esi
    6389:	89 f5                	mov    %esi,%ebp
    638b:	42 c7 44 28 fc ff ff 	movl   $0xffffffff,-0x4(%rax,%r13,1)
    6392:	ff ff 
    6394:	8b 44 24 18          	mov    0x18(%rsp),%eax
    6398:	83 e8 01             	sub    $0x1,%eax
    639b:	89 44 24 5c          	mov    %eax,0x5c(%rsp)
    639f:	90                   	nop
    63a0:	48 63 54 24 1c       	movslq 0x1c(%rsp),%rdx
    63a5:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    63aa:	8b 0c 90             	mov    (%rax,%rdx,4),%ecx
    63ad:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    63b2:	48 63 04 90          	movslq (%rax,%rdx,4),%rax
    63b6:	83 f9 ff             	cmp    $0xffffffff,%ecx
    63b9:	0f 84 62 03 00 00    	je     6721 <jerasure_smart_bitmatrix_to_schedule+0x4e1>
    63bf:	48 8b 5c 24 40       	mov    0x40(%rsp),%rbx
    63c4:	48 63 f1             	movslq %ecx,%rsi
    63c7:	89 04 b3             	mov    %eax,(%rbx,%rsi,4)
    63ca:	83 f8 ff             	cmp    $0xffffffff,%eax
    63cd:	74 08                	je     63d7 <jerasure_smart_bitmatrix_to_schedule+0x197>
    63cf:	48 8b 74 24 60       	mov    0x60(%rsp),%rsi
    63d4:	89 0c 86             	mov    %ecx,(%rsi,%rax,4)
    63d7:	8b 44 24 14          	mov    0x14(%rsp),%eax
    63db:	0f af 44 24 1c       	imul   0x1c(%rsp),%eax
    63e0:	48 8b 5c 24 20       	mov    0x20(%rsp),%rbx
    63e5:	41 0f af c6          	imul   %r14d,%eax
    63e9:	48 98                	cltq   
    63eb:	48 8d 1c 83          	lea    (%rbx,%rax,4),%rbx
    63ef:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    63f4:	44 8b 24 90          	mov    (%rax,%rdx,4),%r12d
    63f8:	48 63 c5             	movslq %ebp,%rax
    63fb:	4c 8d 2c c5 00 00 00 	lea    0x0(,%rax,8),%r13
    6402:	00 
    6403:	4b 8d 0c 2f          	lea    (%r15,%r13,1),%rcx
    6407:	41 83 fc ff          	cmp    $0xffffffff,%r12d
    640b:	0f 85 a6 01 00 00    	jne    65b7 <jerasure_smart_bitmatrix_to_schedule+0x377>
    6411:	44 8b 5c 24 18       	mov    0x18(%rsp),%r11d
    6416:	45 85 db             	test   %r11d,%r11d
    6419:	0f 8e c7 00 00 00    	jle    64e6 <jerasure_smart_bitmatrix_to_schedule+0x2a6>
    641f:	8b 74 24 5c          	mov    0x5c(%rsp),%esi
    6423:	45 31 db             	xor    %r11d,%r11d
    6426:	44 89 f1             	mov    %r14d,%ecx
    6429:	45 31 e4             	xor    %r12d,%r12d
    642c:	44 89 da             	mov    %r11d,%edx
    642f:	41 89 ee             	mov    %ebp,%r14d
    6432:	49 89 f3             	mov    %rsi,%r11
    6435:	48 89 de             	mov    %rbx,%rsi
    6438:	eb 1a                	jmp    6454 <jerasure_smart_bitmatrix_to_schedule+0x214>
    643a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    6440:	49 8d 44 24 01       	lea    0x1(%r12),%rax
    6445:	4d 39 e3             	cmp    %r12,%r11
    6448:	0f 84 8a 00 00 00    	je     64d8 <jerasure_smart_bitmatrix_to_schedule+0x298>
    644e:	49 89 c4             	mov    %rax,%r12
    6451:	49 63 c6             	movslq %r14d,%rax
    6454:	42 8b 3c a6          	mov    (%rsi,%r12,4),%edi
    6458:	4c 8d 2c c5 00 00 00 	lea    0x0(,%rax,8),%r13
    645f:	00 
    6460:	4b 8d 2c 2f          	lea    (%r15,%r13,1),%rbp
    6464:	85 ff                	test   %edi,%edi
    6466:	74 d8                	je     6440 <jerasure_smart_bitmatrix_to_schedule+0x200>
    6468:	bf 14 00 00 00       	mov    $0x14,%edi
    646d:	89 4c 24 48          	mov    %ecx,0x48(%rsp)
    6471:	41 83 c6 01          	add    $0x1,%r14d
    6475:	4c 89 5c 24 38       	mov    %r11,0x38(%rsp)
    647a:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
    647f:	89 54 24 08          	mov    %edx,0x8(%rsp)
    6483:	e8 68 af ff ff       	callq  13f0 <malloc@plt>
    6488:	8b 54 24 08          	mov    0x8(%rsp),%edx
    648c:	8b 4c 24 48          	mov    0x48(%rsp),%ecx
    6490:	48 89 c7             	mov    %rax,%rdi
    6493:	48 89 45 00          	mov    %rax,0x0(%rbp)
    6497:	4c 8b 5c 24 38       	mov    0x38(%rsp),%r11
    649c:	4b 8d 6c 2f 08       	lea    0x8(%r15,%r13,1),%rbp
    64a1:	89 50 10             	mov    %edx,0x10(%rax)
    64a4:	44 89 e0             	mov    %r12d,%eax
    64a7:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
    64ac:	99                   	cltd   
    64ad:	f7 f9                	idiv   %ecx
    64af:	89 07                	mov    %eax,(%rdi)
    64b1:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
    64b5:	89 57 04             	mov    %edx,0x4(%rdi)
    64b8:	99                   	cltd   
    64b9:	f7 f9                	idiv   %ecx
    64bb:	03 44 24 14          	add    0x14(%rsp),%eax
    64bf:	89 47 08             	mov    %eax,0x8(%rdi)
    64c2:	49 8d 44 24 01       	lea    0x1(%r12),%rax
    64c7:	89 57 0c             	mov    %edx,0xc(%rdi)
    64ca:	ba 01 00 00 00       	mov    $0x1,%edx
    64cf:	4d 39 e3             	cmp    %r12,%r11
    64d2:	0f 85 76 ff ff ff    	jne    644e <jerasure_smart_bitmatrix_to_schedule+0x20e>
    64d8:	89 c8                	mov    %ecx,%eax
    64da:	48 89 f3             	mov    %rsi,%rbx
    64dd:	48 89 e9             	mov    %rbp,%rcx
    64e0:	44 89 f5             	mov    %r14d,%ebp
    64e3:	41 89 c6             	mov    %eax,%r14d
    64e6:	8b 74 24 68          	mov    0x68(%rsp),%esi
    64ea:	83 fe ff             	cmp    $0xffffffff,%esi
    64ed:	0f 84 d7 01 00 00    	je     66ca <jerasure_smart_bitmatrix_to_schedule+0x48a>
    64f3:	44 8b 5c 24 5c       	mov    0x5c(%rsp),%r11d
    64f8:	44 8b 64 24 6c       	mov    0x6c(%rsp),%r12d
    64fd:	89 6c 24 30          	mov    %ebp,0x30(%rsp)
    6501:	44 8b 6c 24 1c       	mov    0x1c(%rsp),%r13d
    6506:	48 8b 6c 24 50       	mov    0x50(%rsp),%rbp
    650b:	4c 89 7c 24 38       	mov    %r15,0x38(%rsp)
    6510:	4c 89 5c 24 08       	mov    %r11,0x8(%rsp)
    6515:	4c 8b 7c 24 40       	mov    0x40(%rsp),%r15
    651a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    6520:	8b 44 24 14          	mov    0x14(%rsp),%eax
    6524:	8b 54 24 18          	mov    0x18(%rsp),%edx
    6528:	0f af c6             	imul   %esi,%eax
    652b:	41 0f af c6          	imul   %r14d,%eax
    652f:	48 98                	cltq   
    6531:	85 d2                	test   %edx,%edx
    6533:	7e 7b                	jle    65b0 <jerasure_smart_bitmatrix_to_schedule+0x370>
    6535:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    653a:	4c 8b 5c 24 08       	mov    0x8(%rsp),%r11
    653f:	b9 01 00 00 00       	mov    $0x1,%ecx
    6544:	48 8d 3c 87          	lea    (%rdi,%rax,4),%rdi
    6548:	31 c0                	xor    %eax,%eax
    654a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    6550:	8b 14 83             	mov    (%rbx,%rax,4),%edx
    6553:	33 14 87             	xor    (%rdi,%rax,4),%edx
    6556:	01 d1                	add    %edx,%ecx
    6558:	48 89 c2             	mov    %rax,%rdx
    655b:	48 83 c0 01          	add    $0x1,%rax
    655f:	49 39 d3             	cmp    %rdx,%r11
    6562:	75 ec                	jne    6550 <jerasure_smart_bitmatrix_to_schedule+0x310>
    6564:	4c 89 5c 24 08       	mov    %r11,0x8(%rsp)
    6569:	48 63 d6             	movslq %esi,%rdx
    656c:	48 8d 7c 95 00       	lea    0x0(%rbp,%rdx,4),%rdi
    6571:	8b 07                	mov    (%rdi),%eax
    6573:	39 c8                	cmp    %ecx,%eax
    6575:	7e 12                	jle    6589 <jerasure_smart_bitmatrix_to_schedule+0x349>
    6577:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    657c:	44 8b 5c 24 1c       	mov    0x1c(%rsp),%r11d
    6581:	89 0f                	mov    %ecx,(%rdi)
    6583:	44 89 1c 90          	mov    %r11d,(%rax,%rdx,4)
    6587:	89 c8                	mov    %ecx,%eax
    6589:	41 39 c4             	cmp    %eax,%r12d
    658c:	7e 06                	jle    6594 <jerasure_smart_bitmatrix_to_schedule+0x354>
    658e:	41 89 c4             	mov    %eax,%r12d
    6591:	41 89 f5             	mov    %esi,%r13d
    6594:	41 8b 34 97          	mov    (%r15,%rdx,4),%esi
    6598:	83 fe ff             	cmp    $0xffffffff,%esi
    659b:	75 83                	jne    6520 <jerasure_smart_bitmatrix_to_schedule+0x2e0>
    659d:	8b 6c 24 30          	mov    0x30(%rsp),%ebp
    65a1:	4c 8b 7c 24 38       	mov    0x38(%rsp),%r15
    65a6:	44 89 6c 24 1c       	mov    %r13d,0x1c(%rsp)
    65ab:	e9 f0 fd ff ff       	jmpq   63a0 <jerasure_smart_bitmatrix_to_schedule+0x160>
    65b0:	b9 01 00 00 00       	mov    $0x1,%ecx
    65b5:	eb b2                	jmp    6569 <jerasure_smart_bitmatrix_to_schedule+0x329>
    65b7:	bf 14 00 00 00       	mov    $0x14,%edi
    65bc:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
    65c1:	83 c5 01             	add    $0x1,%ebp
    65c4:	e8 27 ae ff ff       	callq  13f0 <malloc@plt>
    65c9:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    65ce:	c7 40 10 00 00 00 00 	movl   $0x0,0x10(%rax)
    65d5:	48 89 c6             	mov    %rax,%rsi
    65d8:	48 89 01             	mov    %rax,(%rcx)
    65db:	44 89 e0             	mov    %r12d,%eax
    65de:	8b 4c 24 14          	mov    0x14(%rsp),%ecx
    65e2:	99                   	cltd   
    65e3:	41 f7 fe             	idiv   %r14d
    65e6:	44 0f af e1          	imul   %ecx,%r12d
    65ea:	45 0f af e6          	imul   %r14d,%r12d
    65ee:	4d 63 e4             	movslq %r12d,%r12
    65f1:	01 c8                	add    %ecx,%eax
    65f3:	89 56 04             	mov    %edx,0x4(%rsi)
    65f6:	89 06                	mov    %eax,(%rsi)
    65f8:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
    65fc:	99                   	cltd   
    65fd:	41 f7 fe             	idiv   %r14d
    6600:	01 c8                	add    %ecx,%eax
    6602:	8b 4c 24 18          	mov    0x18(%rsp),%ecx
    6606:	89 54 24 58          	mov    %edx,0x58(%rsp)
    660a:	89 44 24 4c          	mov    %eax,0x4c(%rsp)
    660e:	89 46 08             	mov    %eax,0x8(%rsi)
    6611:	89 56 0c             	mov    %edx,0xc(%rsi)
    6614:	85 c9                	test   %ecx,%ecx
    6616:	0f 8e 2e 01 00 00    	jle    674a <jerasure_smart_bitmatrix_to_schedule+0x50a>
    661c:	8b 44 24 5c          	mov    0x5c(%rsp),%eax
    6620:	44 89 74 24 48       	mov    %r14d,0x48(%rsp)
    6625:	49 89 de             	mov    %rbx,%r14
    6628:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    662d:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    6632:	4e 8d 2c a0          	lea    (%rax,%r12,4),%r13
    6636:	45 31 e4             	xor    %r12d,%r12d
    6639:	4c 89 e9             	mov    %r13,%rcx
    663c:	eb 05                	jmp    6643 <jerasure_smart_bitmatrix_to_schedule+0x403>
    663e:	66 90                	xchg   %ax,%ax
    6640:	49 89 c4             	mov    %rax,%r12
    6643:	4c 63 ed             	movslq %ebp,%r13
    6646:	42 8b 04 a1          	mov    (%rcx,%r12,4),%eax
    664a:	49 c1 e5 03          	shl    $0x3,%r13
    664e:	4b 8d 14 2f          	lea    (%r15,%r13,1),%rdx
    6652:	43 39 04 a6          	cmp    %eax,(%r14,%r12,4)
    6656:	74 4e                	je     66a6 <jerasure_smart_bitmatrix_to_schedule+0x466>
    6658:	bf 14 00 00 00       	mov    $0x14,%edi
    665d:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
    6662:	83 c5 01             	add    $0x1,%ebp
    6665:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
    666a:	e8 81 ad ff ff       	callq  13f0 <malloc@plt>
    666f:	48 8b 54 24 38       	mov    0x38(%rsp),%rdx
    6674:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    6679:	c7 40 10 01 00 00 00 	movl   $0x1,0x10(%rax)
    6680:	48 89 c6             	mov    %rax,%rsi
    6683:	48 89 02             	mov    %rax,(%rdx)
    6686:	44 89 e0             	mov    %r12d,%eax
    6689:	99                   	cltd   
    668a:	f7 7c 24 48          	idivl  0x48(%rsp)
    668e:	89 06                	mov    %eax,(%rsi)
    6690:	8b 44 24 4c          	mov    0x4c(%rsp),%eax
    6694:	89 56 04             	mov    %edx,0x4(%rsi)
    6697:	4b 8d 54 2f 08       	lea    0x8(%r15,%r13,1),%rdx
    669c:	89 46 08             	mov    %eax,0x8(%rsi)
    669f:	8b 44 24 58          	mov    0x58(%rsp),%eax
    66a3:	89 46 0c             	mov    %eax,0xc(%rsi)
    66a6:	49 8d 44 24 01       	lea    0x1(%r12),%rax
    66ab:	4c 39 64 24 08       	cmp    %r12,0x8(%rsp)
    66b0:	75 8e                	jne    6640 <jerasure_smart_bitmatrix_to_schedule+0x400>
    66b2:	8b 74 24 68          	mov    0x68(%rsp),%esi
    66b6:	4c 89 f3             	mov    %r14,%rbx
    66b9:	48 89 d1             	mov    %rdx,%rcx
    66bc:	44 8b 74 24 48       	mov    0x48(%rsp),%r14d
    66c1:	83 fe ff             	cmp    $0xffffffff,%esi
    66c4:	0f 85 29 fe ff ff    	jne    64f3 <jerasure_smart_bitmatrix_to_schedule+0x2b3>
    66ca:	bf 14 00 00 00       	mov    $0x14,%edi
    66cf:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
    66d4:	e8 17 ad ff ff       	callq  13f0 <malloc@plt>
    66d9:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    66de:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    66e3:	c7 00 ff ff ff ff    	movl   $0xffffffff,(%rax)
    66e9:	48 89 01             	mov    %rax,(%rcx)
    66ec:	e8 7f ab ff ff       	callq  1270 <free@plt>
    66f1:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
    66f6:	e8 75 ab ff ff       	callq  1270 <free@plt>
    66fb:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
    6700:	e8 6b ab ff ff       	callq  1270 <free@plt>
    6705:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    670a:	e8 61 ab ff ff       	callq  1270 <free@plt>
    670f:	48 83 c4 78          	add    $0x78,%rsp
    6713:	4c 89 f8             	mov    %r15,%rax
    6716:	5b                   	pop    %rbx
    6717:	5d                   	pop    %rbp
    6718:	41 5c                	pop    %r12
    671a:	41 5d                	pop    %r13
    671c:	41 5e                	pop    %r14
    671e:	41 5f                	pop    %r15
    6720:	c3                   	retq   
    6721:	c7 44 24 68 ff ff ff 	movl   $0xffffffff,0x68(%rsp)
    6728:	ff 
    6729:	83 f8 ff             	cmp    $0xffffffff,%eax
    672c:	0f 84 a5 fc ff ff    	je     63d7 <jerasure_smart_bitmatrix_to_schedule+0x197>
    6732:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
    6737:	48 63 c8             	movslq %eax,%rcx
    673a:	89 44 24 68          	mov    %eax,0x68(%rsp)
    673e:	c7 04 8b ff ff ff ff 	movl   $0xffffffff,(%rbx,%rcx,4)
    6745:	e9 8d fc ff ff       	jmpq   63d7 <jerasure_smart_bitmatrix_to_schedule+0x197>
    674a:	4b 8d 4c 2f 08       	lea    0x8(%r15,%r13,1),%rcx
    674f:	e9 92 fd ff ff       	jmpq   64e6 <jerasure_smart_bitmatrix_to_schedule+0x2a6>
    6754:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    675b:	00 00 00 00 
    675f:	90                   	nop

0000000000006760 <jerasure_generate_decoding_schedule>:
    6760:	41 57                	push   %r15
    6762:	4d 89 c7             	mov    %r8,%r15
    6765:	41 56                	push   %r14
    6767:	41 55                	push   %r13
    6769:	41 54                	push   %r12
    676b:	41 89 f4             	mov    %esi,%r12d
    676e:	55                   	push   %rbp
    676f:	53                   	push   %rbx
    6770:	48 81 ec b8 00 00 00 	sub    $0xb8,%rsp
    6777:	41 8b 00             	mov    (%r8),%eax
    677a:	89 7c 24 3c          	mov    %edi,0x3c(%rsp)
    677e:	89 54 24 38          	mov    %edx,0x38(%rsp)
    6782:	48 89 4c 24 78       	mov    %rcx,0x78(%rsp)
    6787:	44 89 8c 24 ac 00 00 	mov    %r9d,0xac(%rsp)
    678e:	00 
    678f:	83 f8 ff             	cmp    $0xffffffff,%eax
    6792:	0f 84 96 06 00 00    	je     6e2e <jerasure_generate_decoding_schedule+0x6ce>
    6798:	49 8d 50 04          	lea    0x4(%r8),%rdx
    679c:	45 31 ed             	xor    %r13d,%r13d
    679f:	31 ed                	xor    %ebp,%ebp
    67a1:	eb 0e                	jmp    67b1 <jerasure_generate_decoding_schedule+0x51>
    67a3:	8b 02                	mov    (%rdx),%eax
    67a5:	48 83 c2 04          	add    $0x4,%rdx
    67a9:	83 c5 01             	add    $0x1,%ebp
    67ac:	83 f8 ff             	cmp    $0xffffffff,%eax
    67af:	74 15                	je     67c6 <jerasure_generate_decoding_schedule+0x66>
    67b1:	39 44 24 3c          	cmp    %eax,0x3c(%rsp)
    67b5:	7f ec                	jg     67a3 <jerasure_generate_decoding_schedule+0x43>
    67b7:	8b 02                	mov    (%rdx),%eax
    67b9:	48 83 c2 04          	add    $0x4,%rdx
    67bd:	41 83 c5 01          	add    $0x1,%r13d
    67c1:	83 f8 ff             	cmp    $0xffffffff,%eax
    67c4:	75 eb                	jne    67b1 <jerasure_generate_decoding_schedule+0x51>
    67c6:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
    67ca:	46 8d 34 20          	lea    (%rax,%r12,1),%r14d
    67ce:	49 63 c6             	movslq %r14d,%rax
    67d1:	48 8d 1c 85 00 00 00 	lea    0x0(,%rax,4),%rbx
    67d8:	00 
    67d9:	48 89 df             	mov    %rbx,%rdi
    67dc:	e8 0f ac ff ff       	callq  13f0 <malloc@plt>
    67e1:	48 89 df             	mov    %rbx,%rdi
    67e4:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    67e9:	e8 02 ac ff ff       	callq  13f0 <malloc@plt>
    67ee:	8b 5c 24 3c          	mov    0x3c(%rsp),%ebx
    67f2:	44 89 e6             	mov    %r12d,%esi
    67f5:	4c 89 fa             	mov    %r15,%rdx
    67f8:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    67fd:	89 df                	mov    %ebx,%edi
    67ff:	e8 9c dc ff ff       	callq  44a0 <jerasure_erasures_to_erased>
    6804:	49 89 c4             	mov    %rax,%r12
    6807:	48 85 c0             	test   %rax,%rax
    680a:	0f 84 4a 04 00 00    	je     6c5a <jerasure_generate_decoding_schedule+0x4fa>
    6810:	85 db                	test   %ebx,%ebx
    6812:	0f 8e 0d 06 00 00    	jle    6e25 <jerasure_generate_decoding_schedule+0x6c5>
    6818:	89 de                	mov    %ebx,%esi
    681a:	44 8d 7b ff          	lea    -0x1(%rbx),%r15d
    681e:	89 d9                	mov    %ebx,%ecx
    6820:	31 d2                	xor    %edx,%edx
    6822:	eb 1c                	jmp    6840 <jerasure_generate_decoding_schedule+0xe0>
    6824:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    6829:	89 14 90             	mov    %edx,(%rax,%rdx,4)
    682c:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
    6831:	89 14 90             	mov    %edx,(%rax,%rdx,4)
    6834:	48 8d 42 01          	lea    0x1(%rdx),%rax
    6838:	4c 39 fa             	cmp    %r15,%rdx
    683b:	74 6c                	je     68a9 <jerasure_generate_decoding_schedule+0x149>
    683d:	48 89 c2             	mov    %rax,%rdx
    6840:	41 8b 04 94          	mov    (%r12,%rdx,4),%eax
    6844:	41 89 d3             	mov    %edx,%r11d
    6847:	85 c0                	test   %eax,%eax
    6849:	74 d9                	je     6824 <jerasure_generate_decoding_schedule+0xc4>
    684b:	48 63 d9             	movslq %ecx,%rbx
    684e:	8d 41 01             	lea    0x1(%rcx),%eax
    6851:	48 8d 3c 9d 00 00 00 	lea    0x0(,%rbx,4),%rdi
    6858:	00 
    6859:	41 8b 1c 9c          	mov    (%r12,%rbx,4),%ebx
    685d:	48 98                	cltq   
    685f:	85 db                	test   %ebx,%ebx
    6861:	74 17                	je     687a <jerasure_generate_decoding_schedule+0x11a>
    6863:	89 c1                	mov    %eax,%ecx
    6865:	48 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%rdi
    686c:	00 
    686d:	48 83 c0 01          	add    $0x1,%rax
    6871:	41 8b 5c 84 fc       	mov    -0x4(%r12,%rax,4),%ebx
    6876:	85 db                	test   %ebx,%ebx
    6878:	75 e9                	jne    6863 <jerasure_generate_decoding_schedule+0x103>
    687a:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
    687f:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
    6884:	44 89 1c 38          	mov    %r11d,(%rax,%rdi,1)
    6888:	48 63 c6             	movslq %esi,%rax
    688b:	89 0c 93             	mov    %ecx,(%rbx,%rdx,4)
    688e:	83 c1 01             	add    $0x1,%ecx
    6891:	44 89 1c 83          	mov    %r11d,(%rbx,%rax,4)
    6895:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
    689a:	89 34 90             	mov    %esi,(%rax,%rdx,4)
    689d:	83 c6 01             	add    $0x1,%esi
    68a0:	48 8d 42 01          	lea    0x1(%rdx),%rax
    68a4:	4c 39 fa             	cmp    %r15,%rdx
    68a7:	75 94                	jne    683d <jerasure_generate_decoding_schedule+0xdd>
    68a9:	48 63 44 24 3c       	movslq 0x3c(%rsp),%rax
    68ae:	41 39 c6             	cmp    %eax,%r14d
    68b1:	7e 27                	jle    68da <jerasure_generate_decoding_schedule+0x17a>
    68b3:	41 8b 3c 84          	mov    (%r12,%rax,4),%edi
    68b7:	85 ff                	test   %edi,%edi
    68b9:	74 16                	je     68d1 <jerasure_generate_decoding_schedule+0x171>
    68bb:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
    68c0:	48 63 d6             	movslq %esi,%rdx
    68c3:	89 04 93             	mov    %eax,(%rbx,%rdx,4)
    68c6:	48 8b 5c 24 48       	mov    0x48(%rsp),%rbx
    68cb:	89 34 83             	mov    %esi,(%rbx,%rax,4)
    68ce:	83 c6 01             	add    $0x1,%esi
    68d1:	48 83 c0 01          	add    $0x1,%rax
    68d5:	41 39 c6             	cmp    %eax,%r14d
    68d8:	7f d9                	jg     68b3 <jerasure_generate_decoding_schedule+0x153>
    68da:	4c 89 e7             	mov    %r12,%rdi
    68dd:	41 8d 5c 2d 00       	lea    0x0(%r13,%rbp,1),%ebx
    68e2:	e8 89 a9 ff ff       	callq  1270 <free@plt>
    68e7:	8b 44 24 38          	mov    0x38(%rsp),%eax
    68eb:	8b 7c 24 3c          	mov    0x3c(%rsp),%edi
    68ef:	89 9c 24 a8 00 00 00 	mov    %ebx,0xa8(%rsp)
    68f6:	0f af f8             	imul   %eax,%edi
    68f9:	89 7c 24 6c          	mov    %edi,0x6c(%rsp)
    68fd:	0f af fb             	imul   %ebx,%edi
    6900:	0f af f8             	imul   %eax,%edi
    6903:	48 63 ff             	movslq %edi,%rdi
    6906:	48 c1 e7 02          	shl    $0x2,%rdi
    690a:	e8 e1 aa ff ff       	callq  13f0 <malloc@plt>
    690f:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    6914:	85 ed                	test   %ebp,%ebp
    6916:	0f 85 71 03 00 00    	jne    6c8d <jerasure_generate_decoding_schedule+0x52d>
    691c:	45 85 ed             	test   %r13d,%r13d
    691f:	0f 84 f0 02 00 00    	je     6c15 <jerasure_generate_decoding_schedule+0x4b5>
    6925:	48 63 4c 24 38       	movslq 0x38(%rsp),%rcx
    692a:	48 63 54 24 3c       	movslq 0x3c(%rsp),%rdx
    692f:	8b 7c 24 6c          	mov    0x6c(%rsp),%edi
    6933:	48 89 ce             	mov    %rcx,%rsi
    6936:	48 89 cb             	mov    %rcx,%rbx
    6939:	49 89 d6             	mov    %rdx,%r14
    693c:	48 0f af f1          	imul   %rcx,%rsi
    6940:	89 f8                	mov    %edi,%eax
    6942:	0f af c1             	imul   %ecx,%eax
    6945:	48 0f af f2          	imul   %rdx,%rsi
    6949:	48 c1 e6 02          	shl    $0x2,%rsi
    694d:	48 89 b4 24 80 00 00 	mov    %rsi,0x80(%rsp)
    6954:	00 
    6955:	48 8d 34 8d 00 00 00 	lea    0x0(,%rcx,4),%rsi
    695c:	00 
    695d:	48 63 cd             	movslq %ebp,%rcx
    6960:	48 89 b4 24 88 00 00 	mov    %rsi,0x88(%rsp)
    6967:	00 
    6968:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
    696d:	48 01 ca             	add    %rcx,%rdx
    6970:	48 8d 0c 96          	lea    (%rsi,%rdx,4),%rcx
    6974:	48 89 4c 24 70       	mov    %rcx,0x70(%rsp)
    6979:	48 63 c8             	movslq %eax,%rcx
    697c:	0f af c5             	imul   %ebp,%eax
    697f:	48 c1 e1 02          	shl    $0x2,%rcx
    6983:	48 89 8c 24 a0 00 00 	mov    %rcx,0xa0(%rsp)
    698a:	00 
    698b:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    6990:	48 98                	cltq   
    6992:	48 8d 0c 81          	lea    (%rcx,%rax,4),%rcx
    6996:	8d 47 ff             	lea    -0x1(%rdi),%eax
    6999:	89 df                	mov    %ebx,%edi
    699b:	41 0f af de          	imul   %r14d,%ebx
    699f:	48 8d 44 81 04       	lea    0x4(%rcx,%rax,4),%rax
    69a4:	48 89 4c 24 40       	mov    %rcx,0x40(%rsp)
    69a9:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    69ae:	41 8d 45 ff          	lea    -0x1(%r13),%eax
    69b2:	48 01 c2             	add    %rax,%rdx
    69b5:	48 8d 44 96 04       	lea    0x4(%rsi,%rdx,4),%rax
    69ba:	48 89 84 24 98 00 00 	mov    %rax,0x98(%rsp)
    69c1:	00 
    69c2:	48 63 c3             	movslq %ebx,%rax
    69c5:	4c 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%r15
    69cc:	00 
    69cd:	8d 47 ff             	lea    -0x1(%rdi),%eax
    69d0:	48 89 84 24 90 00 00 	mov    %rax,0x90(%rsp)
    69d7:	00 
    69d8:	48 f7 d0             	not    %rax
    69db:	48 c1 e0 02          	shl    $0x2,%rax
    69df:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    69e4:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
    69e9:	44 8b 74 24 3c       	mov    0x3c(%rsp),%r14d
    69ee:	8b 7c 24 38          	mov    0x38(%rsp),%edi
    69f2:	48 8b 94 24 80 00 00 	mov    0x80(%rsp),%rdx
    69f9:	00 
    69fa:	8b 00                	mov    (%rax),%eax
    69fc:	44 29 f0             	sub    %r14d,%eax
    69ff:	41 0f af c6          	imul   %r14d,%eax
    6a03:	0f af c7             	imul   %edi,%eax
    6a06:	0f af c7             	imul   %edi,%eax
    6a09:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
    6a0e:	48 98                	cltq   
    6a10:	48 8d 34 87          	lea    (%rdi,%rax,4),%rsi
    6a14:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    6a19:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    6a1e:	e8 8d a9 ff ff       	callq  13b0 <memcpy@plt>
    6a23:	45 85 f6             	test   %r14d,%r14d
    6a26:	0f 8e be 01 00 00    	jle    6bea <jerasure_generate_decoding_schedule+0x48a>
    6a2c:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
    6a30:	89 5c 24 20          	mov    %ebx,0x20(%rsp)
    6a34:	45 31 ed             	xor    %r13d,%r13d
    6a37:	31 ed                	xor    %ebp,%ebp
    6a39:	48 8b 9c 24 88 00 00 	mov    0x88(%rsp),%rbx
    6a40:	00 
    6a41:	44 8b 74 24 38       	mov    0x38(%rsp),%r14d
    6a46:	83 e8 01             	sub    $0x1,%eax
    6a49:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    6a4e:	eb 11                	jmp    6a61 <jerasure_generate_decoding_schedule+0x301>
    6a50:	48 8d 45 01          	lea    0x1(%rbp),%rax
    6a54:	45 01 f5             	add    %r14d,%r13d
    6a57:	48 39 6c 24 30       	cmp    %rbp,0x30(%rsp)
    6a5c:	74 49                	je     6aa7 <jerasure_generate_decoding_schedule+0x347>
    6a5e:	48 89 c5             	mov    %rax,%rbp
    6a61:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    6a66:	39 2c a8             	cmp    %ebp,(%rax,%rbp,4)
    6a69:	74 e5                	je     6a50 <jerasure_generate_decoding_schedule+0x2f0>
    6a6b:	45 85 f6             	test   %r14d,%r14d
    6a6e:	7e e0                	jle    6a50 <jerasure_generate_decoding_schedule+0x2f0>
    6a70:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    6a75:	49 63 c5             	movslq %r13d,%rax
    6a78:	45 31 e4             	xor    %r12d,%r12d
    6a7b:	48 8d 3c 87          	lea    (%rdi,%rax,4),%rdi
    6a7f:	90                   	nop
    6a80:	48 89 da             	mov    %rbx,%rdx
    6a83:	31 f6                	xor    %esi,%esi
    6a85:	41 83 c4 01          	add    $0x1,%r12d
    6a89:	e8 b2 a8 ff ff       	callq  1340 <memset@plt>
    6a8e:	48 89 c7             	mov    %rax,%rdi
    6a91:	4c 01 ff             	add    %r15,%rdi
    6a94:	45 39 e6             	cmp    %r12d,%r14d
    6a97:	75 e7                	jne    6a80 <jerasure_generate_decoding_schedule+0x320>
    6a99:	48 8d 45 01          	lea    0x1(%rbp),%rax
    6a9d:	45 01 f5             	add    %r14d,%r13d
    6aa0:	48 39 6c 24 30       	cmp    %rbp,0x30(%rsp)
    6aa5:	75 b7                	jne    6a5e <jerasure_generate_decoding_schedule+0x2fe>
    6aa7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    6aac:	48 03 84 24 90 00 00 	add    0x90(%rsp),%rax
    6ab3:	00 
    6ab4:	c7 44 24 68 00 00 00 	movl   $0x0,0x68(%rsp)
    6abb:	00 
    6abc:	8b 5c 24 20          	mov    0x20(%rsp),%ebx
    6ac0:	44 8b 74 24 6c       	mov    0x6c(%rsp),%r14d
    6ac5:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    6aca:	48 c7 44 24 20 00 00 	movq   $0x0,0x20(%rsp)
    6ad1:	00 00 
    6ad3:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    6ad8:	48 83 c0 04          	add    $0x4,%rax
    6adc:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    6ae1:	eb 26                	jmp    6b09 <jerasure_generate_decoding_schedule+0x3a9>
    6ae3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    6ae8:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    6aed:	8b 54 24 38          	mov    0x38(%rsp),%edx
    6af1:	01 54 24 68          	add    %edx,0x68(%rsp)
    6af5:	48 8d 47 01          	lea    0x1(%rdi),%rax
    6af9:	48 39 7c 24 30       	cmp    %rdi,0x30(%rsp)
    6afe:	0f 84 e6 00 00 00    	je     6bea <jerasure_generate_decoding_schedule+0x48a>
    6b04:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    6b09:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    6b0e:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    6b13:	39 04 87             	cmp    %eax,(%rdi,%rax,4)
    6b16:	74 d0                	je     6ae8 <jerasure_generate_decoding_schedule+0x388>
    6b18:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
    6b1d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    6b22:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    6b25:	41 89 c4             	mov    %eax,%r12d
    6b28:	89 44 24 08          	mov    %eax,0x8(%rsp)
    6b2c:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
    6b30:	41 29 c4             	sub    %eax,%r12d
    6b33:	44 0f af e0          	imul   %eax,%r12d
    6b37:	8b 44 24 38          	mov    0x38(%rsp),%eax
    6b3b:	44 0f af e0          	imul   %eax,%r12d
    6b3f:	44 0f af e0          	imul   %eax,%r12d
    6b43:	4d 63 e4             	movslq %r12d,%r12
    6b46:	85 c0                	test   %eax,%eax
    6b48:	7e 9e                	jle    6ae8 <jerasure_generate_decoding_schedule+0x388>
    6b4a:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
    6b4f:	48 63 44 24 68       	movslq 0x68(%rsp),%rax
    6b54:	45 31 ed             	xor    %r13d,%r13d
    6b57:	48 03 44 24 58       	add    0x58(%rsp),%rax
    6b5c:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
    6b61:	48 8b 6c 24 40       	mov    0x40(%rsp),%rbp
    6b66:	4c 8d 1c 87          	lea    (%rdi,%rax,4),%r11
    6b6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    6b70:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    6b75:	31 ff                	xor    %edi,%edi
    6b77:	4a 8d 0c 18          	lea    (%rax,%r11,1),%rcx
    6b7b:	eb 0e                	jmp    6b8b <jerasure_generate_decoding_schedule+0x42b>
    6b7d:	0f 1f 00             	nopl   (%rax)
    6b80:	48 83 c1 04          	add    $0x4,%rcx
    6b84:	01 df                	add    %ebx,%edi
    6b86:	49 39 cb             	cmp    %rcx,%r11
    6b89:	74 46                	je     6bd1 <jerasure_generate_decoding_schedule+0x471>
    6b8b:	8b 01                	mov    (%rcx),%eax
    6b8d:	85 c0                	test   %eax,%eax
    6b8f:	74 ef                	je     6b80 <jerasure_generate_decoding_schedule+0x420>
    6b91:	45 85 f6             	test   %r14d,%r14d
    6b94:	7e ea                	jle    6b80 <jerasure_generate_decoding_schedule+0x420>
    6b96:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    6b9b:	48 63 c7             	movslq %edi,%rax
    6b9e:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
    6ba3:	4c 01 e0             	add    %r12,%rax
    6ba6:	48 8d 14 82          	lea    (%rdx,%rax,4),%rdx
    6baa:	48 89 e8             	mov    %rbp,%rax
    6bad:	0f 1f 00             	nopl   (%rax)
    6bb0:	8b 0a                	mov    (%rdx),%ecx
    6bb2:	31 08                	xor    %ecx,(%rax)
    6bb4:	48 83 c0 04          	add    $0x4,%rax
    6bb8:	48 83 c2 04          	add    $0x4,%rdx
    6bbc:	48 39 c6             	cmp    %rax,%rsi
    6bbf:	75 ef                	jne    6bb0 <jerasure_generate_decoding_schedule+0x450>
    6bc1:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    6bc6:	01 df                	add    %ebx,%edi
    6bc8:	48 83 c1 04          	add    $0x4,%rcx
    6bcc:	49 39 cb             	cmp    %rcx,%r11
    6bcf:	75 ba                	jne    6b8b <jerasure_generate_decoding_schedule+0x42b>
    6bd1:	41 83 c5 01          	add    $0x1,%r13d
    6bd5:	4d 01 fb             	add    %r15,%r11
    6bd8:	4c 01 fd             	add    %r15,%rbp
    6bdb:	4c 01 fe             	add    %r15,%rsi
    6bde:	44 39 6c 24 38       	cmp    %r13d,0x38(%rsp)
    6be3:	75 8b                	jne    6b70 <jerasure_generate_decoding_schedule+0x410>
    6be5:	e9 fe fe ff ff       	jmpq   6ae8 <jerasure_generate_decoding_schedule+0x388>
    6bea:	48 8b b4 24 a0 00 00 	mov    0xa0(%rsp),%rsi
    6bf1:	00 
    6bf2:	48 83 44 24 70 04    	addq   $0x4,0x70(%rsp)
    6bf8:	48 01 74 24 40       	add    %rsi,0x40(%rsp)
    6bfd:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
    6c02:	48 01 74 24 50       	add    %rsi,0x50(%rsp)
    6c07:	48 39 84 24 98 00 00 	cmp    %rax,0x98(%rsp)
    6c0e:	00 
    6c0f:	0f 85 cf fd ff ff    	jne    69e4 <jerasure_generate_decoding_schedule+0x284>
    6c15:	8b 94 24 ac 00 00 00 	mov    0xac(%rsp),%edx
    6c1c:	85 d2                	test   %edx,%edx
    6c1e:	75 4f                	jne    6c6f <jerasure_generate_decoding_schedule+0x50f>
    6c20:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    6c25:	8b 54 24 38          	mov    0x38(%rsp),%edx
    6c29:	8b b4 24 a8 00 00 00 	mov    0xa8(%rsp),%esi
    6c30:	8b 7c 24 3c          	mov    0x3c(%rsp),%edi
    6c34:	e8 77 f4 ff ff       	callq  60b0 <jerasure_dumb_bitmatrix_to_schedule>
    6c39:	49 89 c4             	mov    %rax,%r12
    6c3c:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    6c41:	e8 2a a6 ff ff       	callq  1270 <free@plt>
    6c46:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
    6c4b:	e8 20 a6 ff ff       	callq  1270 <free@plt>
    6c50:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    6c55:	e8 16 a6 ff ff       	callq  1270 <free@plt>
    6c5a:	48 81 c4 b8 00 00 00 	add    $0xb8,%rsp
    6c61:	4c 89 e0             	mov    %r12,%rax
    6c64:	5b                   	pop    %rbx
    6c65:	5d                   	pop    %rbp
    6c66:	41 5c                	pop    %r12
    6c68:	41 5d                	pop    %r13
    6c6a:	41 5e                	pop    %r14
    6c6c:	41 5f                	pop    %r15
    6c6e:	c3                   	retq   
    6c6f:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    6c74:	8b 54 24 38          	mov    0x38(%rsp),%edx
    6c78:	8b b4 24 a8 00 00 00 	mov    0xa8(%rsp),%esi
    6c7f:	8b 7c 24 3c          	mov    0x3c(%rsp),%edi
    6c83:	e8 b8 f5 ff ff       	callq  6240 <jerasure_smart_bitmatrix_to_schedule>
    6c88:	49 89 c4             	mov    %rax,%r12
    6c8b:	eb af                	jmp    6c3c <jerasure_generate_decoding_schedule+0x4dc>
    6c8d:	8b 5c 24 3c          	mov    0x3c(%rsp),%ebx
    6c91:	8b 44 24 38          	mov    0x38(%rsp),%eax
    6c95:	0f af c3             	imul   %ebx,%eax
    6c98:	0f af c0             	imul   %eax,%eax
    6c9b:	4c 63 f0             	movslq %eax,%r14
    6c9e:	49 c1 e6 02          	shl    $0x2,%r14
    6ca2:	4c 89 f7             	mov    %r14,%rdi
    6ca5:	e8 46 a7 ff ff       	callq  13f0 <malloc@plt>
    6caa:	49 89 c4             	mov    %rax,%r12
    6cad:	85 db                	test   %ebx,%ebx
    6caf:	0f 8e c8 00 00 00    	jle    6d7d <jerasure_generate_decoding_schedule+0x61d>
    6cb5:	48 63 44 24 6c       	movslq 0x6c(%rsp),%rax
    6cba:	8b 5c 24 38          	mov    0x38(%rsp),%ebx
    6cbe:	c7 44 24 08 00 00 00 	movl   $0x0,0x8(%rsp)
    6cc5:	00 
    6cc6:	4c 89 e7             	mov    %r12,%rdi
    6cc9:	0f af d8             	imul   %eax,%ebx
    6ccc:	48 8d 04 85 04 00 00 	lea    0x4(,%rax,4),%rax
    6cd3:	00 
    6cd4:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    6cd9:	48 63 d3             	movslq %ebx,%rdx
    6cdc:	89 5c 24 18          	mov    %ebx,0x18(%rsp)
    6ce0:	8b 5c 24 3c          	mov    0x3c(%rsp),%ebx
    6ce4:	4c 8d 3c 95 00 00 00 	lea    0x0(,%rdx,4),%r15
    6ceb:	00 
    6cec:	83 eb 01             	sub    $0x1,%ebx
    6cef:	48 89 5c 24 30       	mov    %rbx,0x30(%rsp)
    6cf4:	31 db                	xor    %ebx,%ebx
    6cf6:	48 89 d8             	mov    %rbx,%rax
    6cf9:	4c 89 fb             	mov    %r15,%rbx
    6cfc:	49 89 c7             	mov    %rax,%r15
    6cff:	eb 38                	jmp    6d39 <jerasure_generate_decoding_schedule+0x5d9>
    6d01:	2b 44 24 3c          	sub    0x3c(%rsp),%eax
    6d05:	0f af 44 24 18       	imul   0x18(%rsp),%eax
    6d0a:	48 89 da             	mov    %rbx,%rdx
    6d0d:	48 8b 74 24 78       	mov    0x78(%rsp),%rsi
    6d12:	48 98                	cltq   
    6d14:	48 8d 34 86          	lea    (%rsi,%rax,4),%rsi
    6d18:	e8 93 a6 ff ff       	callq  13b0 <memcpy@plt>
    6d1d:	48 89 c7             	mov    %rax,%rdi
    6d20:	8b 54 24 38          	mov    0x38(%rsp),%edx
    6d24:	48 01 df             	add    %rbx,%rdi
    6d27:	01 54 24 08          	add    %edx,0x8(%rsp)
    6d2b:	49 8d 47 01          	lea    0x1(%r15),%rax
    6d2f:	4c 3b 7c 24 30       	cmp    0x30(%rsp),%r15
    6d34:	74 47                	je     6d7d <jerasure_generate_decoding_schedule+0x61d>
    6d36:	49 89 c7             	mov    %rax,%r15
    6d39:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    6d3e:	42 8b 04 b8          	mov    (%rax,%r15,4),%eax
    6d42:	44 39 f8             	cmp    %r15d,%eax
    6d45:	75 ba                	jne    6d01 <jerasure_generate_decoding_schedule+0x5a1>
    6d47:	48 89 da             	mov    %rbx,%rdx
    6d4a:	31 f6                	xor    %esi,%esi
    6d4c:	e8 ef a5 ff ff       	callq  1340 <memset@plt>
    6d51:	8b 4c 24 38          	mov    0x38(%rsp),%ecx
    6d55:	48 89 c7             	mov    %rax,%rdi
    6d58:	85 c9                	test   %ecx,%ecx
    6d5a:	7e c4                	jle    6d20 <jerasure_generate_decoding_schedule+0x5c0>
    6d5c:	48 63 44 24 08       	movslq 0x8(%rsp),%rax
    6d61:	48 8d 0c 87          	lea    (%rdi,%rax,4),%rcx
    6d65:	31 c0                	xor    %eax,%eax
    6d67:	c7 01 01 00 00 00    	movl   $0x1,(%rcx)
    6d6d:	83 c0 01             	add    $0x1,%eax
    6d70:	48 03 4c 24 20       	add    0x20(%rsp),%rcx
    6d75:	39 44 24 38          	cmp    %eax,0x38(%rsp)
    6d79:	75 ec                	jne    6d67 <jerasure_generate_decoding_schedule+0x607>
    6d7b:	eb a3                	jmp    6d20 <jerasure_generate_decoding_schedule+0x5c0>
    6d7d:	4c 89 f7             	mov    %r14,%rdi
    6d80:	e8 6b a6 ff ff       	callq  13f0 <malloc@plt>
    6d85:	8b 5c 24 6c          	mov    0x6c(%rsp),%ebx
    6d89:	4c 89 e7             	mov    %r12,%rdi
    6d8c:	48 89 c6             	mov    %rax,%rsi
    6d8f:	49 89 c6             	mov    %rax,%r14
    6d92:	89 da                	mov    %ebx,%edx
    6d94:	e8 27 e2 ff ff       	callq  4fc0 <jerasure_invert_bitmatrix>
    6d99:	4c 89 e7             	mov    %r12,%rdi
    6d9c:	e8 cf a4 ff ff       	callq  1270 <free@plt>
    6da1:	48 63 44 24 38       	movslq 0x38(%rsp),%rax
    6da6:	48 63 4c 24 3c       	movslq 0x3c(%rsp),%rcx
    6dab:	48 89 c7             	mov    %rax,%rdi
    6dae:	48 0f af c0          	imul   %rax,%rax
    6db2:	0f af df             	imul   %edi,%ebx
    6db5:	48 0f af c1          	imul   %rcx,%rax
    6db9:	48 89 c2             	mov    %rax,%rdx
    6dbc:	48 63 c3             	movslq %ebx,%rax
    6dbf:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
    6dc4:	49 89 c4             	mov    %rax,%r12
    6dc7:	48 c1 e0 02          	shl    $0x2,%rax
    6dcb:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    6dd0:	8d 45 ff             	lea    -0x1(%rbp),%eax
    6dd3:	4c 8d 3c 8b          	lea    (%rbx,%rcx,4),%r15
    6dd7:	48 c1 e2 02          	shl    $0x2,%rdx
    6ddb:	48 01 c1             	add    %rax,%rcx
    6dde:	48 8d 5c 8b 04       	lea    0x4(%rbx,%rcx,4),%rbx
    6de3:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    6de8:	41 8b 07             	mov    (%r15),%eax
    6deb:	48 89 cf             	mov    %rcx,%rdi
    6dee:	48 89 54 24 18       	mov    %rdx,0x18(%rsp)
    6df3:	49 83 c7 04          	add    $0x4,%r15
    6df7:	41 0f af c4          	imul   %r12d,%eax
    6dfb:	48 98                	cltq   
    6dfd:	49 8d 34 86          	lea    (%r14,%rax,4),%rsi
    6e01:	e8 aa a5 ff ff       	callq  13b0 <memcpy@plt>
    6e06:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    6e0b:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
    6e10:	48 01 c1             	add    %rax,%rcx
    6e13:	4c 39 fb             	cmp    %r15,%rbx
    6e16:	75 d0                	jne    6de8 <jerasure_generate_decoding_schedule+0x688>
    6e18:	4c 89 f7             	mov    %r14,%rdi
    6e1b:	e8 50 a4 ff ff       	callq  1270 <free@plt>
    6e20:	e9 f7 fa ff ff       	jmpq   691c <jerasure_generate_decoding_schedule+0x1bc>
    6e25:	8b 74 24 3c          	mov    0x3c(%rsp),%esi
    6e29:	e9 7b fa ff ff       	jmpq   68a9 <jerasure_generate_decoding_schedule+0x149>
    6e2e:	45 31 ed             	xor    %r13d,%r13d
    6e31:	31 ed                	xor    %ebp,%ebp
    6e33:	e9 8e f9 ff ff       	jmpq   67c6 <jerasure_generate_decoding_schedule+0x66>
    6e38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    6e3f:	00 

0000000000006e40 <jerasure_schedule_decode_lazy>:
    6e40:	f3 0f 1e fa          	endbr64 
    6e44:	41 57                	push   %r15
    6e46:	41 56                	push   %r14
    6e48:	4d 89 c6             	mov    %r8,%r14
    6e4b:	41 55                	push   %r13
    6e4d:	49 89 cd             	mov    %rcx,%r13
    6e50:	4c 89 c9             	mov    %r9,%rcx
    6e53:	41 54                	push   %r12
    6e55:	41 89 d4             	mov    %edx,%r12d
    6e58:	4c 89 f2             	mov    %r14,%rdx
    6e5b:	55                   	push   %rbp
    6e5c:	89 fd                	mov    %edi,%ebp
    6e5e:	53                   	push   %rbx
    6e5f:	89 f3                	mov    %esi,%ebx
    6e61:	48 83 ec 18          	sub    $0x18,%rsp
    6e65:	4c 8b 44 24 50       	mov    0x50(%rsp),%r8
    6e6a:	e8 d1 d6 ff ff       	callq  4540 <set_up_ptrs_for_scheduled_decoding>
    6e6f:	48 85 c0             	test   %rax,%rax
    6e72:	0f 84 a1 00 00 00    	je     6f19 <jerasure_schedule_decode_lazy+0xd9>
    6e78:	44 8b 4c 24 68       	mov    0x68(%rsp),%r9d
    6e7d:	4c 89 e9             	mov    %r13,%rcx
    6e80:	4d 89 f0             	mov    %r14,%r8
    6e83:	44 89 e2             	mov    %r12d,%edx
    6e86:	89 de                	mov    %ebx,%esi
    6e88:	89 ef                	mov    %ebp,%edi
    6e8a:	49 89 c7             	mov    %rax,%r15
    6e8d:	e8 ce f8 ff ff       	callq  6760 <jerasure_generate_decoding_schedule>
    6e92:	49 89 c5             	mov    %rax,%r13
    6e95:	48 85 c0             	test   %rax,%rax
    6e98:	0f 84 82 00 00 00    	je     6f20 <jerasure_schedule_decode_lazy+0xe0>
    6e9e:	8b 44 24 58          	mov    0x58(%rsp),%eax
    6ea2:	85 c0                	test   %eax,%eax
    6ea4:	7e 52                	jle    6ef8 <jerasure_schedule_decode_lazy+0xb8>
    6ea6:	44 0f af 64 24 60    	imul   0x60(%rsp),%r12d
    6eac:	01 dd                	add    %ebx,%ebp
    6eae:	8d 55 ff             	lea    -0x1(%rbp),%edx
    6eb1:	49 8d 5c d7 08       	lea    0x8(%r15,%rdx,8),%rbx
    6eb6:	44 89 64 24 0c       	mov    %r12d,0xc(%rsp)
    6ebb:	4d 63 f4             	movslq %r12d,%r14
    6ebe:	45 31 e4             	xor    %r12d,%r12d
    6ec1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    6ec8:	8b 54 24 60          	mov    0x60(%rsp),%edx
    6ecc:	4c 89 ee             	mov    %r13,%rsi
    6ecf:	4c 89 ff             	mov    %r15,%rdi
    6ed2:	e8 79 ee ff ff       	callq  5d50 <jerasure_do_scheduled_operations>
    6ed7:	4c 89 fa             	mov    %r15,%rdx
    6eda:	85 ed                	test   %ebp,%ebp
    6edc:	7e 0e                	jle    6eec <jerasure_schedule_decode_lazy+0xac>
    6ede:	66 90                	xchg   %ax,%ax
    6ee0:	4c 01 32             	add    %r14,(%rdx)
    6ee3:	48 83 c2 08          	add    $0x8,%rdx
    6ee7:	48 39 da             	cmp    %rbx,%rdx
    6eea:	75 f4                	jne    6ee0 <jerasure_schedule_decode_lazy+0xa0>
    6eec:	44 03 64 24 0c       	add    0xc(%rsp),%r12d
    6ef1:	44 39 64 24 58       	cmp    %r12d,0x58(%rsp)
    6ef6:	7f d0                	jg     6ec8 <jerasure_schedule_decode_lazy+0x88>
    6ef8:	4c 89 ef             	mov    %r13,%rdi
    6efb:	e8 70 d7 ff ff       	callq  4670 <jerasure_free_schedule>
    6f00:	4c 89 ff             	mov    %r15,%rdi
    6f03:	e8 68 a3 ff ff       	callq  1270 <free@plt>
    6f08:	31 c0                	xor    %eax,%eax
    6f0a:	48 83 c4 18          	add    $0x18,%rsp
    6f0e:	5b                   	pop    %rbx
    6f0f:	5d                   	pop    %rbp
    6f10:	41 5c                	pop    %r12
    6f12:	41 5d                	pop    %r13
    6f14:	41 5e                	pop    %r14
    6f16:	41 5f                	pop    %r15
    6f18:	c3                   	retq   
    6f19:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    6f1e:	eb ea                	jmp    6f0a <jerasure_schedule_decode_lazy+0xca>
    6f20:	4c 89 ff             	mov    %r15,%rdi
    6f23:	e8 48 a3 ff ff       	callq  1270 <free@plt>
    6f28:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    6f2d:	eb db                	jmp    6f0a <jerasure_schedule_decode_lazy+0xca>
    6f2f:	90                   	nop

0000000000006f30 <jerasure_generate_schedule_cache>:
    6f30:	f3 0f 1e fa          	endbr64 
    6f34:	41 57                	push   %r15
    6f36:	41 56                	push   %r14
    6f38:	41 55                	push   %r13
    6f3a:	41 54                	push   %r12
    6f3c:	55                   	push   %rbp
    6f3d:	53                   	push   %rbx
    6f3e:	48 83 ec 68          	sub    $0x68,%rsp
    6f42:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    6f49:	00 00 
    6f4b:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    6f50:	31 c0                	xor    %eax,%eax
    6f52:	83 fe 02             	cmp    $0x2,%esi
    6f55:	0f 85 55 01 00 00    	jne    70b0 <jerasure_generate_schedule_cache+0x180>
    6f5b:	8d 5f 02             	lea    0x2(%rdi),%ebx
    6f5e:	89 fd                	mov    %edi,%ebp
    6f60:	8d 7f 03             	lea    0x3(%rdi),%edi
    6f63:	41 89 d4             	mov    %edx,%r12d
    6f66:	0f af fb             	imul   %ebx,%edi
    6f69:	49 89 cd             	mov    %rcx,%r13
    6f6c:	45 89 c6             	mov    %r8d,%r14d
    6f6f:	48 63 ff             	movslq %edi,%rdi
    6f72:	48 c1 e7 03          	shl    $0x3,%rdi
    6f76:	e8 75 a4 ff ff       	callq  13f0 <malloc@plt>
    6f7b:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    6f80:	48 85 c0             	test   %rax,%rax
    6f83:	0f 84 27 01 00 00    	je     70b0 <jerasure_generate_schedule_cache+0x180>
    6f89:	85 db                	test   %ebx,%ebx
    6f8b:	0f 8e 28 01 00 00    	jle    70b9 <jerasure_generate_schedule_cache+0x189>
    6f91:	48 63 db             	movslq %ebx,%rbx
    6f94:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
    6f99:	c7 44 24 4c 00 00 00 	movl   $0x0,0x4c(%rsp)
    6fa0:	00 
    6fa1:	48 8d 04 dd 08 00 00 	lea    0x8(,%rbx,8),%rax
    6fa8:	00 
    6fa9:	48 c7 04 24 01 00 00 	movq   $0x1,(%rsp)
    6fb0:	00 
    6fb1:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    6fb6:	48 8d 04 dd 00 00 00 	lea    0x0(,%rbx,8),%rax
    6fbd:	00 
    6fbe:	48 89 c6             	mov    %rax,%rsi
    6fc1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    6fc6:	48 89 c8             	mov    %rcx,%rax
    6fc9:	48 01 f0             	add    %rsi,%rax
    6fcc:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    6fd1:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    6fd6:	8d 45 01             	lea    0x1(%rbp),%eax
    6fd9:	48 83 c0 01          	add    $0x1,%rax
    6fdd:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    6fe2:	48 8d 44 24 4c       	lea    0x4c(%rsp),%rax
    6fe7:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    6fec:	0f 1f 40 00          	nopl   0x0(%rax)
    6ff0:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
    6ff5:	4c 89 e9             	mov    %r13,%rcx
    6ff8:	be 02 00 00 00       	mov    $0x2,%esi
    6ffd:	45 89 f1             	mov    %r14d,%r9d
    7000:	44 89 e2             	mov    %r12d,%edx
    7003:	89 ef                	mov    %ebp,%edi
    7005:	c7 44 24 50 ff ff ff 	movl   $0xffffffff,0x50(%rsp)
    700c:	ff 
    700d:	e8 4e f7 ff ff       	callq  6760 <jerasure_generate_decoding_schedule>
    7012:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    7017:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
    701c:	48 89 01             	mov    %rax,(%rcx)
    701f:	48 39 34 24          	cmp    %rsi,(%rsp)
    7023:	0f 84 90 00 00 00    	je     70b9 <jerasure_generate_schedule_cache+0x189>
    7029:	48 8b 04 24          	mov    (%rsp),%rax
    702d:	89 44 24 4c          	mov    %eax,0x4c(%rsp)
    7031:	85 c0                	test   %eax,%eax
    7033:	7e 54                	jle    7089 <jerasure_generate_schedule_cache+0x159>
    7035:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    703a:	48 8b 3c 24          	mov    (%rsp),%rdi
    703e:	31 db                	xor    %ebx,%ebx
    7040:	4c 8d 3c f8          	lea    (%rax,%rdi,8),%r15
    7044:	0f 1f 40 00          	nopl   0x0(%rax)
    7048:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
    704d:	44 89 e2             	mov    %r12d,%edx
    7050:	45 89 f1             	mov    %r14d,%r9d
    7053:	4c 89 e9             	mov    %r13,%rcx
    7056:	be 02 00 00 00       	mov    $0x2,%esi
    705b:	89 ef                	mov    %ebp,%edi
    705d:	89 5c 24 50          	mov    %ebx,0x50(%rsp)
    7061:	c7 44 24 54 ff ff ff 	movl   $0xffffffff,0x54(%rsp)
    7068:	ff 
    7069:	e8 f2 f6 ff ff       	callq  6760 <jerasure_generate_decoding_schedule>
    706e:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    7073:	48 89 04 da          	mov    %rax,(%rdx,%rbx,8)
    7077:	48 83 c3 01          	add    $0x1,%rbx
    707b:	49 89 07             	mov    %rax,(%r15)
    707e:	4c 03 7c 24 18       	add    0x18(%rsp),%r15
    7083:	48 3b 1c 24          	cmp    (%rsp),%rbx
    7087:	75 bf                	jne    7048 <jerasure_generate_schedule_cache+0x118>
    7089:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    708e:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
    7093:	48 83 04 24 01       	addq   $0x1,(%rsp)
    7098:	48 01 4c 24 20       	add    %rcx,0x20(%rsp)
    709d:	48 01 74 24 10       	add    %rsi,0x10(%rsp)
    70a2:	e9 49 ff ff ff       	jmpq   6ff0 <jerasure_generate_schedule_cache+0xc0>
    70a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    70ae:	00 00 
    70b0:	48 c7 44 24 28 00 00 	movq   $0x0,0x28(%rsp)
    70b7:	00 00 
    70b9:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
    70be:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    70c5:	00 00 
    70c7:	75 14                	jne    70dd <jerasure_generate_schedule_cache+0x1ad>
    70c9:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    70ce:	48 83 c4 68          	add    $0x68,%rsp
    70d2:	5b                   	pop    %rbx
    70d3:	5d                   	pop    %rbp
    70d4:	41 5c                	pop    %r12
    70d6:	41 5d                	pop    %r13
    70d8:	41 5e                	pop    %r14
    70da:	41 5f                	pop    %r15
    70dc:	c3                   	retq   
    70dd:	e8 1e a2 ff ff       	callq  1300 <__stack_chk_fail@plt>
    70e2:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    70e9:	00 00 00 00 
    70ed:	0f 1f 00             	nopl   (%rax)

00000000000070f0 <jerasure_bitmatrix_encode>:
    70f0:	f3 0f 1e fa          	endbr64 
    70f4:	41 57                	push   %r15
    70f6:	41 56                	push   %r14
    70f8:	41 55                	push   %r13
    70fa:	41 54                	push   %r12
    70fc:	55                   	push   %rbp
    70fd:	53                   	push   %rbx
    70fe:	48 83 ec 18          	sub    $0x18,%rsp
    7102:	f6 44 24 58 07       	testb  $0x7,0x58(%rsp)
    7107:	0f 85 9a 00 00 00    	jne    71a7 <jerasure_bitmatrix_encode+0xb7>
    710d:	48 89 cd             	mov    %rcx,%rbp
    7110:	8b 4c 24 58          	mov    0x58(%rsp),%ecx
    7114:	8b 44 24 50          	mov    0x50(%rsp),%eax
    7118:	41 89 d4             	mov    %edx,%r12d
    711b:	0f af ca             	imul   %edx,%ecx
    711e:	99                   	cltd   
    711f:	f7 f9                	idiv   %ecx
    7121:	85 d2                	test   %edx,%edx
    7123:	0f 85 ac 00 00 00    	jne    71d5 <jerasure_bitmatrix_encode+0xe5>
    7129:	85 f6                	test   %esi,%esi
    712b:	7e 6b                	jle    7198 <jerasure_bitmatrix_encode+0xa8>
    712d:	44 89 e0             	mov    %r12d,%eax
    7130:	48 89 eb             	mov    %rbp,%rbx
    7133:	41 89 fd             	mov    %edi,%r13d
    7136:	4d 89 c6             	mov    %r8,%r14
    7139:	0f af c7             	imul   %edi,%eax
    713c:	4d 89 cf             	mov    %r9,%r15
    713f:	89 fd                	mov    %edi,%ebp
    7141:	41 0f af c4          	imul   %r12d,%eax
    7145:	48 98                	cltq   
    7147:	48 c1 e0 02          	shl    $0x2,%rax
    714b:	48 89 04 24          	mov    %rax,(%rsp)
    714f:	8d 04 3e             	lea    (%rsi,%rdi,1),%eax
    7152:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    7156:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    715d:	00 00 00 
    7160:	48 83 ec 08          	sub    $0x8,%rsp
    7164:	41 89 e8             	mov    %ebp,%r8d
    7167:	48 89 da             	mov    %rbx,%rdx
    716a:	4d 89 f1             	mov    %r14,%r9
    716d:	8b 44 24 60          	mov    0x60(%rsp),%eax
    7171:	31 c9                	xor    %ecx,%ecx
    7173:	44 89 e6             	mov    %r12d,%esi
    7176:	44 89 ef             	mov    %r13d,%edi
    7179:	83 c5 01             	add    $0x1,%ebp
    717c:	50                   	push   %rax
    717d:	8b 44 24 60          	mov    0x60(%rsp),%eax
    7181:	50                   	push   %rax
    7182:	41 57                	push   %r15
    7184:	e8 f7 c5 ff ff       	callq  3780 <jerasure_bitmatrix_dotprod>
    7189:	48 03 5c 24 20       	add    0x20(%rsp),%rbx
    718e:	48 83 c4 20          	add    $0x20,%rsp
    7192:	3b 6c 24 0c          	cmp    0xc(%rsp),%ebp
    7196:	75 c8                	jne    7160 <jerasure_bitmatrix_encode+0x70>
    7198:	48 83 c4 18          	add    $0x18,%rsp
    719c:	5b                   	pop    %rbx
    719d:	5d                   	pop    %rbp
    719e:	41 5c                	pop    %r12
    71a0:	41 5d                	pop    %r13
    71a2:	41 5e                	pop    %r14
    71a4:	41 5f                	pop    %r15
    71a6:	c3                   	retq   
    71a7:	48 8b 3d 92 9f 00 00 	mov    0x9f92(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    71ae:	8b 4c 24 58          	mov    0x58(%rsp),%ecx
    71b2:	31 c0                	xor    %eax,%eax
    71b4:	41 b8 25 00 00 00    	mov    $0x25,%r8d
    71ba:	48 8d 15 27 36 00 00 	lea    0x3627(%rip),%rdx        # a7e8 <__PRETTY_FUNCTION__.5230+0x117>
    71c1:	be 01 00 00 00       	mov    $0x1,%esi
    71c6:	e8 a5 a2 ff ff       	callq  1470 <__fprintf_chk@plt>
    71cb:	bf 01 00 00 00       	mov    $0x1,%edi
    71d0:	e8 7b a2 ff ff       	callq  1450 <exit@plt>
    71d5:	50                   	push   %rax
    71d6:	48 8b 3d 63 9f 00 00 	mov    0x9f63(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    71dd:	31 c0                	xor    %eax,%eax
    71df:	41 b8 25 00 00 00    	mov    $0x25,%r8d
    71e5:	41 54                	push   %r12
    71e7:	44 8b 4c 24 68       	mov    0x68(%rsp),%r9d
    71ec:	48 8d 15 3d 36 00 00 	lea    0x363d(%rip),%rdx        # a830 <__PRETTY_FUNCTION__.5230+0x15f>
    71f3:	be 01 00 00 00       	mov    $0x1,%esi
    71f8:	8b 4c 24 60          	mov    0x60(%rsp),%ecx
    71fc:	e8 6f a2 ff ff       	callq  1470 <__fprintf_chk@plt>
    7201:	bf 01 00 00 00       	mov    $0x1,%edi
    7206:	e8 45 a2 ff ff       	callq  1450 <exit@plt>
    720b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000007210 <galois_create_log_tables.part.0>:
    7210:	41 57                	push   %r15
    7212:	48 8d 05 87 39 00 00 	lea    0x3987(%rip),%rax        # aba0 <nw>
    7219:	41 56                	push   %r14
    721b:	4c 8d 35 fe a2 00 00 	lea    0xa2fe(%rip),%r14        # 11520 <galois_log_tables>
    7222:	41 55                	push   %r13
    7224:	41 54                	push   %r12
    7226:	4c 63 e7             	movslq %edi,%r12
    7229:	55                   	push   %rbp
    722a:	53                   	push   %rbx
    722b:	48 83 ec 08          	sub    $0x8,%rsp
    722f:	4e 63 2c a0          	movslq (%rax,%r12,4),%r13
    7233:	4a 8d 3c ad 00 00 00 	lea    0x0(,%r13,4),%rdi
    723a:	00 
    723b:	e8 b0 a1 ff ff       	callq  13f0 <malloc@plt>
    7240:	4b 89 04 e6          	mov    %rax,(%r14,%r12,8)
    7244:	48 85 c0             	test   %rax,%rax
    7247:	0f 84 3c 01 00 00    	je     7389 <galois_create_log_tables.part.0+0x179>
    724d:	4b 8d 7c 6d 00       	lea    0x0(%r13,%r13,2),%rdi
    7252:	4c 89 eb             	mov    %r13,%rbx
    7255:	48 89 c5             	mov    %rax,%rbp
    7258:	48 c1 e7 02          	shl    $0x2,%rdi
    725c:	4c 8d 2d 9d a1 00 00 	lea    0xa19d(%rip),%r13        # 11400 <galois_ilog_tables>
    7263:	e8 88 a1 ff ff       	callq  13f0 <malloc@plt>
    7268:	4b 89 44 e5 00       	mov    %rax,0x0(%r13,%r12,8)
    726d:	48 85 c0             	test   %rax,%rax
    7270:	0f 84 fe 00 00 00    	je     7374 <galois_create_log_tables.part.0+0x164>
    7276:	48 8d 15 83 38 00 00 	lea    0x3883(%rip),%rdx        # ab00 <nwm1>
    727d:	8d 7b ff             	lea    -0x1(%rbx),%edi
    7280:	42 8b 34 a2          	mov    (%rdx,%r12,4),%esi
    7284:	31 d2                	xor    %edx,%edx
    7286:	85 db                	test   %ebx,%ebx
    7288:	7e 1d                	jle    72a7 <galois_create_log_tables.part.0+0x97>
    728a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7290:	48 89 d1             	mov    %rdx,%rcx
    7293:	89 74 95 00          	mov    %esi,0x0(%rbp,%rdx,4)
    7297:	c7 04 90 00 00 00 00 	movl   $0x0,(%rax,%rdx,4)
    729e:	48 83 c2 01          	add    $0x1,%rdx
    72a2:	48 39 cf             	cmp    %rcx,%rdi
    72a5:	75 e9                	jne    7290 <galois_create_log_tables.part.0+0x80>
    72a7:	48 63 ce             	movslq %esi,%rcx
    72aa:	85 f6                	test   %esi,%esi
    72ac:	7e 75                	jle    7323 <galois_create_log_tables.part.0+0x113>
    72ae:	48 89 c2             	mov    %rax,%rdx
    72b1:	48 89 c7             	mov    %rax,%rdi
    72b4:	31 c9                	xor    %ecx,%ecx
    72b6:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    72bc:	4c 8d 3d 1d 3a 00 00 	lea    0x3a1d(%rip),%r15        # ace0 <prim_poly>
    72c3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    72c8:	4d 63 c8             	movslq %r8d,%r9
    72cb:	4e 8d 5c 8d 00       	lea    0x0(%rbp,%r9,4),%r11
    72d0:	47 8d 0c 00          	lea    (%r8,%r8,1),%r9d
    72d4:	45 8b 33             	mov    (%r11),%r14d
    72d7:	41 39 f6             	cmp    %esi,%r14d
    72da:	75 61                	jne    733d <galois_create_log_tables.part.0+0x12d>
    72dc:	44 89 07             	mov    %r8d,(%rdi)
    72df:	45 89 c8             	mov    %r9d,%r8d
    72e2:	41 89 0b             	mov    %ecx,(%r11)
    72e5:	44 85 cb             	test   %r9d,%ebx
    72e8:	74 07                	je     72f1 <galois_create_log_tables.part.0+0xe1>
    72ea:	47 33 04 a7          	xor    (%r15,%r12,4),%r8d
    72ee:	41 21 f0             	and    %esi,%r8d
    72f1:	83 c1 01             	add    $0x1,%ecx
    72f4:	48 83 c7 04          	add    $0x4,%rdi
    72f8:	39 f1                	cmp    %esi,%ecx
    72fa:	75 cc                	jne    72c8 <galois_create_log_tables.part.0+0xb8>
    72fc:	8d 3c 09             	lea    (%rcx,%rcx,1),%edi
    72ff:	8d 71 ff             	lea    -0x1(%rcx),%esi
    7302:	48 63 c9             	movslq %ecx,%rcx
    7305:	4c 8d 44 b0 04       	lea    0x4(%rax,%rsi,4),%r8
    730a:	48 63 ff             	movslq %edi,%rdi
    730d:	0f 1f 00             	nopl   (%rax)
    7310:	8b 32                	mov    (%rdx),%esi
    7312:	89 34 8a             	mov    %esi,(%rdx,%rcx,4)
    7315:	8b 32                	mov    (%rdx),%esi
    7317:	89 34 ba             	mov    %esi,(%rdx,%rdi,4)
    731a:	48 83 c2 04          	add    $0x4,%rdx
    731e:	49 39 d0             	cmp    %rdx,%r8
    7321:	75 ed                	jne    7310 <galois_create_log_tables.part.0+0x100>
    7323:	48 8d 04 88          	lea    (%rax,%rcx,4),%rax
    7327:	4b 89 44 e5 00       	mov    %rax,0x0(%r13,%r12,8)
    732c:	31 c0                	xor    %eax,%eax
    732e:	48 83 c4 08          	add    $0x8,%rsp
    7332:	5b                   	pop    %rbx
    7333:	5d                   	pop    %rbp
    7334:	41 5c                	pop    %r12
    7336:	41 5d                	pop    %r13
    7338:	41 5e                	pop    %r14
    733a:	41 5f                	pop    %r15
    733c:	c3                   	retq   
    733d:	48 8d 05 9c 39 00 00 	lea    0x399c(%rip),%rax        # ace0 <prim_poly>
    7344:	48 8d 15 2d 35 00 00 	lea    0x352d(%rip),%rdx        # a878 <__PRETTY_FUNCTION__.5230+0x1a7>
    734b:	be 01 00 00 00       	mov    $0x1,%esi
    7350:	46 33 0c a0          	xor    (%rax,%r12,4),%r9d
    7354:	41 51                	push   %r9
    7356:	8b 07                	mov    (%rdi),%eax
    7358:	45 89 f1             	mov    %r14d,%r9d
    735b:	48 8b 3d de 9d 00 00 	mov    0x9dde(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    7362:	50                   	push   %rax
    7363:	31 c0                	xor    %eax,%eax
    7365:	e8 06 a1 ff ff       	callq  1470 <__fprintf_chk@plt>
    736a:	bf 01 00 00 00       	mov    $0x1,%edi
    736f:	e8 dc a0 ff ff       	callq  1450 <exit@plt>
    7374:	48 89 ef             	mov    %rbp,%rdi
    7377:	e8 f4 9e ff ff       	callq  1270 <free@plt>
    737c:	4b c7 04 e6 00 00 00 	movq   $0x0,(%r14,%r12,8)
    7383:	00 
    7384:	83 c8 ff             	or     $0xffffffff,%eax
    7387:	eb a5                	jmp    732e <galois_create_log_tables.part.0+0x11e>
    7389:	83 c8 ff             	or     $0xffffffff,%eax
    738c:	eb a0                	jmp    732e <galois_create_log_tables.part.0+0x11e>
    738e:	66 90                	xchg   %ax,%ax

0000000000007390 <galois_create_log_tables>:
    7390:	f3 0f 1e fa          	endbr64 
    7394:	83 ff 1e             	cmp    $0x1e,%edi
    7397:	7f 27                	jg     73c0 <galois_create_log_tables+0x30>
    7399:	48 63 d7             	movslq %edi,%rdx
    739c:	48 8d 05 7d a1 00 00 	lea    0xa17d(%rip),%rax        # 11520 <galois_log_tables>
    73a3:	45 31 c0             	xor    %r8d,%r8d
    73a6:	48 83 3c d0 00       	cmpq   $0x0,(%rax,%rdx,8)
    73ab:	74 0b                	je     73b8 <galois_create_log_tables+0x28>
    73ad:	44 89 c0             	mov    %r8d,%eax
    73b0:	c3                   	retq   
    73b1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    73b8:	e9 53 fe ff ff       	jmpq   7210 <galois_create_log_tables.part.0>
    73bd:	0f 1f 00             	nopl   (%rax)
    73c0:	41 b8 ff ff ff ff    	mov    $0xffffffff,%r8d
    73c6:	eb e5                	jmp    73ad <galois_create_log_tables+0x1d>
    73c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    73cf:	00 

00000000000073d0 <galois_logtable_multiply>:
    73d0:	f3 0f 1e fa          	endbr64 
    73d4:	85 ff                	test   %edi,%edi
    73d6:	74 38                	je     7410 <galois_logtable_multiply+0x40>
    73d8:	85 f6                	test   %esi,%esi
    73da:	74 34                	je     7410 <galois_logtable_multiply+0x40>
    73dc:	48 63 d2             	movslq %edx,%rdx
    73df:	48 8d 05 3a a1 00 00 	lea    0xa13a(%rip),%rax        # 11520 <galois_log_tables>
    73e6:	48 63 ff             	movslq %edi,%rdi
    73e9:	48 63 f6             	movslq %esi,%rsi
    73ec:	48 8b 0c d0          	mov    (%rax,%rdx,8),%rcx
    73f0:	8b 04 b1             	mov    (%rcx,%rsi,4),%eax
    73f3:	03 04 b9             	add    (%rcx,%rdi,4),%eax
    73f6:	48 8d 0d 03 a0 00 00 	lea    0xa003(%rip),%rcx        # 11400 <galois_ilog_tables>
    73fd:	48 8b 14 d1          	mov    (%rcx,%rdx,8),%rdx
    7401:	48 98                	cltq   
    7403:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    7406:	c3                   	retq   
    7407:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    740e:	00 00 
    7410:	31 c0                	xor    %eax,%eax
    7412:	c3                   	retq   
    7413:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    741a:	00 00 00 00 
    741e:	66 90                	xchg   %ax,%ax

0000000000007420 <galois_logtable_divide>:
    7420:	f3 0f 1e fa          	endbr64 
    7424:	85 f6                	test   %esi,%esi
    7426:	74 38                	je     7460 <galois_logtable_divide+0x40>
    7428:	31 c0                	xor    %eax,%eax
    742a:	85 ff                	test   %edi,%edi
    742c:	74 37                	je     7465 <galois_logtable_divide+0x45>
    742e:	48 63 d2             	movslq %edx,%rdx
    7431:	48 8d 05 e8 a0 00 00 	lea    0xa0e8(%rip),%rax        # 11520 <galois_log_tables>
    7438:	48 63 ff             	movslq %edi,%rdi
    743b:	48 63 f6             	movslq %esi,%rsi
    743e:	48 8b 0c d0          	mov    (%rax,%rdx,8),%rcx
    7442:	8b 04 b9             	mov    (%rcx,%rdi,4),%eax
    7445:	2b 04 b1             	sub    (%rcx,%rsi,4),%eax
    7448:	48 8d 0d b1 9f 00 00 	lea    0x9fb1(%rip),%rcx        # 11400 <galois_ilog_tables>
    744f:	48 8b 14 d1          	mov    (%rcx,%rdx,8),%rdx
    7453:	48 98                	cltq   
    7455:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    7458:	c3                   	retq   
    7459:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7460:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    7465:	c3                   	retq   
    7466:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    746d:	00 00 00 

0000000000007470 <galois_create_mult_tables>:
    7470:	f3 0f 1e fa          	endbr64 
    7474:	83 ff 0d             	cmp    $0xd,%edi
    7477:	0f 8f eb 01 00 00    	jg     7668 <galois_create_mult_tables+0x1f8>
    747d:	41 57                	push   %r15
    747f:	4c 63 ff             	movslq %edi,%r15
    7482:	41 56                	push   %r14
    7484:	4c 8d 35 55 9e 00 00 	lea    0x9e55(%rip),%r14        # 112e0 <galois_mult_tables>
    748b:	41 55                	push   %r13
    748d:	41 54                	push   %r12
    748f:	55                   	push   %rbp
    7490:	89 fd                	mov    %edi,%ebp
    7492:	53                   	push   %rbx
    7493:	48 83 ec 18          	sub    $0x18,%rsp
    7497:	4b 83 3c fe 00       	cmpq   $0x0,(%r14,%r15,8)
    749c:	74 12                	je     74b0 <galois_create_mult_tables+0x40>
    749e:	31 c0                	xor    %eax,%eax
    74a0:	48 83 c4 18          	add    $0x18,%rsp
    74a4:	5b                   	pop    %rbx
    74a5:	5d                   	pop    %rbp
    74a6:	41 5c                	pop    %r12
    74a8:	41 5d                	pop    %r13
    74aa:	41 5e                	pop    %r14
    74ac:	41 5f                	pop    %r15
    74ae:	c3                   	retq   
    74af:	90                   	nop
    74b0:	48 8d 05 e9 36 00 00 	lea    0x36e9(%rip),%rax        # aba0 <nw>
    74b7:	4e 63 0c b8          	movslq (%rax,%r15,4),%r9
    74bb:	4d 89 cd             	mov    %r9,%r13
    74be:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
    74c3:	4c 89 cb             	mov    %r9,%rbx
    74c6:	4d 0f af e9          	imul   %r9,%r13
    74ca:	49 c1 e5 02          	shl    $0x2,%r13
    74ce:	4c 89 ef             	mov    %r13,%rdi
    74d1:	e8 1a 9f ff ff       	callq  13f0 <malloc@plt>
    74d6:	4b 89 04 fe          	mov    %rax,(%r14,%r15,8)
    74da:	49 89 c4             	mov    %rax,%r12
    74dd:	48 85 c0             	test   %rax,%rax
    74e0:	0f 84 40 01 00 00    	je     7626 <galois_create_mult_tables+0x1b6>
    74e6:	4c 89 ef             	mov    %r13,%rdi
    74e9:	e8 02 9f ff ff       	callq  13f0 <malloc@plt>
    74ee:	48 8d 15 cb 9c 00 00 	lea    0x9ccb(%rip),%rdx        # 111c0 <galois_div_tables>
    74f5:	4c 8b 4c 24 08       	mov    0x8(%rsp),%r9
    74fa:	48 85 c0             	test   %rax,%rax
    74fd:	4a 89 04 fa          	mov    %rax,(%rdx,%r15,8)
    7501:	49 89 c5             	mov    %rax,%r13
    7504:	0f 84 0c 01 00 00    	je     7616 <galois_create_mult_tables+0x1a6>
    750a:	48 8d 0d 0f a0 00 00 	lea    0xa00f(%rip),%rcx        # 11520 <galois_log_tables>
    7511:	4a 83 3c f9 00       	cmpq   $0x0,(%rcx,%r15,8)
    7516:	0f 84 ca 00 00 00    	je     75e6 <galois_create_mult_tables+0x176>
    751c:	41 c7 04 24 00 00 00 	movl   $0x0,(%r12)
    7523:	00 
    7524:	41 c7 45 00 ff ff ff 	movl   $0xffffffff,0x0(%r13)
    752b:	ff 
    752c:	83 fb 01             	cmp    $0x1,%ebx
    752f:	0f 8e 69 ff ff ff    	jle    749e <galois_create_mult_tables+0x2e>
    7535:	8d 73 fe             	lea    -0x2(%rbx),%esi
    7538:	ba 01 00 00 00       	mov    $0x1,%edx
    753d:	48 8d 46 02          	lea    0x2(%rsi),%rax
    7541:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7548:	41 c7 04 94 00 00 00 	movl   $0x0,(%r12,%rdx,4)
    754f:	00 
    7550:	41 c7 44 95 00 00 00 	movl   $0x0,0x0(%r13,%rdx,4)
    7557:	00 00 
    7559:	48 83 c2 01          	add    $0x1,%rdx
    755d:	48 39 c2             	cmp    %rax,%rdx
    7560:	75 e6                	jne    7548 <galois_create_mult_tables+0xd8>
    7562:	4a 8b 3c f9          	mov    (%rcx,%r15,8),%rdi
    7566:	48 8d 15 93 9e 00 00 	lea    0x9e93(%rip),%rdx        # 11400 <galois_ilog_tables>
    756d:	44 8d 73 ff          	lea    -0x1(%rbx),%r14d
    7571:	4e 8b 04 fa          	mov    (%rdx,%r15,8),%r8
    7575:	48 8d 6f 04          	lea    0x4(%rdi),%rbp
    7579:	4c 8d 7c b7 08       	lea    0x8(%rdi,%rsi,4),%r15
    757e:	66 90                	xchg   %ax,%ax
    7580:	49 c1 e1 02          	shl    $0x2,%r9
    7584:	83 c3 01             	add    $0x1,%ebx
    7587:	ba 01 00 00 00       	mov    $0x1,%edx
    758c:	4f 8d 1c 0c          	lea    (%r12,%r9,1),%r11
    7590:	4d 01 e9             	add    %r13,%r9
    7593:	41 c7 03 00 00 00 00 	movl   $0x0,(%r11)
    759a:	41 c7 01 ff ff ff ff 	movl   $0xffffffff,(%r9)
    75a1:	8b 75 00             	mov    0x0(%rbp),%esi
    75a4:	0f 1f 40 00          	nopl   0x0(%rax)
    75a8:	8b 0c 97             	mov    (%rdi,%rdx,4),%ecx
    75ab:	01 f1                	add    %esi,%ecx
    75ad:	48 63 c9             	movslq %ecx,%rcx
    75b0:	41 8b 0c 88          	mov    (%r8,%rcx,4),%ecx
    75b4:	41 89 0c 93          	mov    %ecx,(%r11,%rdx,4)
    75b8:	89 f1                	mov    %esi,%ecx
    75ba:	2b 0c 97             	sub    (%rdi,%rdx,4),%ecx
    75bd:	48 63 c9             	movslq %ecx,%rcx
    75c0:	41 8b 0c 88          	mov    (%r8,%rcx,4),%ecx
    75c4:	41 89 0c 91          	mov    %ecx,(%r9,%rdx,4)
    75c8:	48 83 c2 01          	add    $0x1,%rdx
    75cc:	48 39 c2             	cmp    %rax,%rdx
    75cf:	75 d7                	jne    75a8 <galois_create_mult_tables+0x138>
    75d1:	48 83 c5 04          	add    $0x4,%rbp
    75d5:	44 01 f3             	add    %r14d,%ebx
    75d8:	49 39 ef             	cmp    %rbp,%r15
    75db:	0f 84 bd fe ff ff    	je     749e <galois_create_mult_tables+0x2e>
    75e1:	4c 63 cb             	movslq %ebx,%r9
    75e4:	eb 9a                	jmp    7580 <galois_create_mult_tables+0x110>
    75e6:	89 ef                	mov    %ebp,%edi
    75e8:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
    75ed:	e8 1e fc ff ff       	callq  7210 <galois_create_log_tables.part.0>
    75f2:	48 8d 15 c7 9b 00 00 	lea    0x9bc7(%rip),%rdx        # 111c0 <galois_div_tables>
    75f9:	85 c0                	test   %eax,%eax
    75fb:	78 33                	js     7630 <galois_create_mult_tables+0x1c0>
    75fd:	4f 8b 24 fe          	mov    (%r14,%r15,8),%r12
    7601:	4e 8b 2c fa          	mov    (%rdx,%r15,8),%r13
    7605:	48 8d 0d 14 9f 00 00 	lea    0x9f14(%rip),%rcx        # 11520 <galois_log_tables>
    760c:	4c 8b 4c 24 08       	mov    0x8(%rsp),%r9
    7611:	e9 06 ff ff ff       	jmpq   751c <galois_create_mult_tables+0xac>
    7616:	4c 89 e7             	mov    %r12,%rdi
    7619:	e8 52 9c ff ff       	callq  1270 <free@plt>
    761e:	4b c7 04 fe 00 00 00 	movq   $0x0,(%r14,%r15,8)
    7625:	00 
    7626:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    762b:	e9 70 fe ff ff       	jmpq   74a0 <galois_create_mult_tables+0x30>
    7630:	4b 8b 3c fe          	mov    (%r14,%r15,8),%rdi
    7634:	e8 37 9c ff ff       	callq  1270 <free@plt>
    7639:	48 8d 15 80 9b 00 00 	lea    0x9b80(%rip),%rdx        # 111c0 <galois_div_tables>
    7640:	4a 8b 3c fa          	mov    (%rdx,%r15,8),%rdi
    7644:	e8 27 9c ff ff       	callq  1270 <free@plt>
    7649:	48 8d 15 70 9b 00 00 	lea    0x9b70(%rip),%rdx        # 111c0 <galois_div_tables>
    7650:	4b c7 04 fe 00 00 00 	movq   $0x0,(%r14,%r15,8)
    7657:	00 
    7658:	83 c8 ff             	or     $0xffffffff,%eax
    765b:	4a c7 04 fa 00 00 00 	movq   $0x0,(%rdx,%r15,8)
    7662:	00 
    7663:	e9 38 fe ff ff       	jmpq   74a0 <galois_create_mult_tables+0x30>
    7668:	83 c8 ff             	or     $0xffffffff,%eax
    766b:	c3                   	retq   
    766c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000007670 <galois_ilog>:
    7670:	f3 0f 1e fa          	endbr64 
    7674:	41 54                	push   %r12
    7676:	4c 8d 25 83 9d 00 00 	lea    0x9d83(%rip),%r12        # 11400 <galois_ilog_tables>
    767d:	55                   	push   %rbp
    767e:	48 63 ee             	movslq %esi,%rbp
    7681:	49 8b 04 ec          	mov    (%r12,%rbp,8),%rax
    7685:	53                   	push   %rbx
    7686:	89 fb                	mov    %edi,%ebx
    7688:	48 85 c0             	test   %rax,%rax
    768b:	74 13                	je     76a0 <galois_ilog+0x30>
    768d:	48 63 fb             	movslq %ebx,%rdi
    7690:	5b                   	pop    %rbx
    7691:	5d                   	pop    %rbp
    7692:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    7695:	41 5c                	pop    %r12
    7697:	c3                   	retq   
    7698:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    769f:	00 
    76a0:	83 fe 1e             	cmp    $0x1e,%esi
    76a3:	7f 1f                	jg     76c4 <galois_ilog+0x54>
    76a5:	48 8d 15 74 9e 00 00 	lea    0x9e74(%rip),%rdx        # 11520 <galois_log_tables>
    76ac:	48 83 3c ea 00       	cmpq   $0x0,(%rdx,%rbp,8)
    76b1:	75 da                	jne    768d <galois_ilog+0x1d>
    76b3:	89 f7                	mov    %esi,%edi
    76b5:	e8 56 fb ff ff       	callq  7210 <galois_create_log_tables.part.0>
    76ba:	85 c0                	test   %eax,%eax
    76bc:	78 06                	js     76c4 <galois_ilog+0x54>
    76be:	49 8b 04 ec          	mov    (%r12,%rbp,8),%rax
    76c2:	eb c9                	jmp    768d <galois_ilog+0x1d>
    76c4:	48 8b 0d 75 9a 00 00 	mov    0x9a75(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    76cb:	ba 2a 00 00 00       	mov    $0x2a,%edx
    76d0:	be 01 00 00 00       	mov    $0x1,%esi
    76d5:	48 8d 3d ec 31 00 00 	lea    0x31ec(%rip),%rdi        # a8c8 <__PRETTY_FUNCTION__.5230+0x1f7>
    76dc:	e8 7f 9d ff ff       	callq  1460 <fwrite@plt>
    76e1:	bf 01 00 00 00       	mov    $0x1,%edi
    76e6:	e8 65 9d ff ff       	callq  1450 <exit@plt>
    76eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000076f0 <galois_log>:
    76f0:	f3 0f 1e fa          	endbr64 
    76f4:	41 54                	push   %r12
    76f6:	4c 63 e6             	movslq %esi,%r12
    76f9:	55                   	push   %rbp
    76fa:	48 8d 2d 1f 9e 00 00 	lea    0x9e1f(%rip),%rbp        # 11520 <galois_log_tables>
    7701:	4a 8b 44 e5 00       	mov    0x0(%rbp,%r12,8),%rax
    7706:	53                   	push   %rbx
    7707:	89 fb                	mov    %edi,%ebx
    7709:	48 85 c0             	test   %rax,%rax
    770c:	74 12                	je     7720 <galois_log+0x30>
    770e:	48 63 fb             	movslq %ebx,%rdi
    7711:	5b                   	pop    %rbx
    7712:	5d                   	pop    %rbp
    7713:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    7716:	41 5c                	pop    %r12
    7718:	c3                   	retq   
    7719:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7720:	83 fe 1e             	cmp    $0x1e,%esi
    7723:	7f 1b                	jg     7740 <galois_log+0x50>
    7725:	89 f7                	mov    %esi,%edi
    7727:	e8 e4 fa ff ff       	callq  7210 <galois_create_log_tables.part.0>
    772c:	85 c0                	test   %eax,%eax
    772e:	78 10                	js     7740 <galois_log+0x50>
    7730:	4a 8b 44 e5 00       	mov    0x0(%rbp,%r12,8),%rax
    7735:	48 63 fb             	movslq %ebx,%rdi
    7738:	5b                   	pop    %rbx
    7739:	5d                   	pop    %rbp
    773a:	41 5c                	pop    %r12
    773c:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    773f:	c3                   	retq   
    7740:	48 8b 0d f9 99 00 00 	mov    0x99f9(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    7747:	ba 29 00 00 00       	mov    $0x29,%edx
    774c:	be 01 00 00 00       	mov    $0x1,%esi
    7751:	48 8d 3d a0 31 00 00 	lea    0x31a0(%rip),%rdi        # a8f8 <__PRETTY_FUNCTION__.5230+0x227>
    7758:	e8 03 9d ff ff       	callq  1460 <fwrite@plt>
    775d:	bf 01 00 00 00       	mov    $0x1,%edi
    7762:	e8 e9 9c ff ff       	callq  1450 <exit@plt>
    7767:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    776e:	00 00 

0000000000007770 <galois_shift_multiply>:
    7770:	f3 0f 1e fa          	endbr64 
    7774:	41 55                	push   %r13
    7776:	41 54                	push   %r12
    7778:	55                   	push   %rbp
    7779:	53                   	push   %rbx
    777a:	48 81 ec 98 00 00 00 	sub    $0x98,%rsp
    7781:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    7788:	00 00 
    778a:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
    7791:	00 
    7792:	31 c0                	xor    %eax,%eax
    7794:	85 d2                	test   %edx,%edx
    7796:	0f 8e cb 00 00 00    	jle    7867 <galois_shift_multiply+0xf7>
    779c:	8d 5a ff             	lea    -0x1(%rdx),%ebx
    779f:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    77a5:	48 89 e5             	mov    %rsp,%rbp
    77a8:	4c 63 da             	movslq %edx,%r11
    77ab:	48 89 d9             	mov    %rbx,%rcx
    77ae:	48 89 e8             	mov    %rbp,%rax
    77b1:	4c 8d 4c 9c 04       	lea    0x4(%rsp,%rbx,4),%r9
    77b6:	41 d3 e0             	shl    %cl,%r8d
    77b9:	4c 8d 2d 20 35 00 00 	lea    0x3520(%rip),%r13        # ace0 <prim_poly>
    77c0:	4c 8d 25 39 33 00 00 	lea    0x3339(%rip),%r12        # ab00 <nwm1>
    77c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    77ce:	00 00 
    77d0:	44 89 c1             	mov    %r8d,%ecx
    77d3:	89 30                	mov    %esi,(%rax)
    77d5:	21 f1                	and    %esi,%ecx
    77d7:	01 f6                	add    %esi,%esi
    77d9:	85 c9                	test   %ecx,%ecx
    77db:	74 09                	je     77e6 <galois_shift_multiply+0x76>
    77dd:	43 33 74 9d 00       	xor    0x0(%r13,%r11,4),%esi
    77e2:	43 23 34 9c          	and    (%r12,%r11,4),%esi
    77e6:	48 83 c0 04          	add    $0x4,%rax
    77ea:	4c 39 c8             	cmp    %r9,%rax
    77ed:	75 e1                	jne    77d0 <galois_shift_multiply+0x60>
    77ef:	31 c9                	xor    %ecx,%ecx
    77f1:	45 31 c0             	xor    %r8d,%r8d
    77f4:	41 bc 01 00 00 00    	mov    $0x1,%r12d
    77fa:	eb 10                	jmp    780c <galois_shift_multiply+0x9c>
    77fc:	0f 1f 40 00          	nopl   0x0(%rax)
    7800:	48 8d 41 01          	lea    0x1(%rcx),%rax
    7804:	48 39 d9             	cmp    %rbx,%rcx
    7807:	74 3a                	je     7843 <galois_shift_multiply+0xd3>
    7809:	48 89 c1             	mov    %rax,%rcx
    780c:	44 89 e0             	mov    %r12d,%eax
    780f:	d3 e0                	shl    %cl,%eax
    7811:	85 f8                	test   %edi,%eax
    7813:	74 eb                	je     7800 <galois_shift_multiply+0x90>
    7815:	44 8b 5c 8d 00       	mov    0x0(%rbp,%rcx,4),%r11d
    781a:	31 f6                	xor    %esi,%esi
    781c:	b8 01 00 00 00       	mov    $0x1,%eax
    7821:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7828:	45 89 d9             	mov    %r11d,%r9d
    782b:	83 c6 01             	add    $0x1,%esi
    782e:	41 21 c1             	and    %eax,%r9d
    7831:	01 c0                	add    %eax,%eax
    7833:	45 31 c8             	xor    %r9d,%r8d
    7836:	39 f2                	cmp    %esi,%edx
    7838:	75 ee                	jne    7828 <galois_shift_multiply+0xb8>
    783a:	48 8d 41 01          	lea    0x1(%rcx),%rax
    783e:	48 39 d9             	cmp    %rbx,%rcx
    7841:	75 c6                	jne    7809 <galois_shift_multiply+0x99>
    7843:	48 8b 84 24 88 00 00 	mov    0x88(%rsp),%rax
    784a:	00 
    784b:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    7852:	00 00 
    7854:	75 16                	jne    786c <galois_shift_multiply+0xfc>
    7856:	48 81 c4 98 00 00 00 	add    $0x98,%rsp
    785d:	44 89 c0             	mov    %r8d,%eax
    7860:	5b                   	pop    %rbx
    7861:	5d                   	pop    %rbp
    7862:	41 5c                	pop    %r12
    7864:	41 5d                	pop    %r13
    7866:	c3                   	retq   
    7867:	45 31 c0             	xor    %r8d,%r8d
    786a:	eb d7                	jmp    7843 <galois_shift_multiply+0xd3>
    786c:	e8 8f 9a ff ff       	callq  1300 <__stack_chk_fail@plt>
    7871:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    7878:	00 00 00 00 
    787c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000007880 <galois_multtable_multiply>:
    7880:	f3 0f 1e fa          	endbr64 
    7884:	48 63 d2             	movslq %edx,%rdx
    7887:	48 8d 05 52 9a 00 00 	lea    0x9a52(%rip),%rax        # 112e0 <galois_mult_tables>
    788e:	48 89 d1             	mov    %rdx,%rcx
    7891:	48 8b 04 d0          	mov    (%rax,%rdx,8),%rax
    7895:	d3 e7                	shl    %cl,%edi
    7897:	09 f7                	or     %esi,%edi
    7899:	48 63 ff             	movslq %edi,%rdi
    789c:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    789f:	c3                   	retq   

00000000000078a0 <galois_multtable_divide>:
    78a0:	f3 0f 1e fa          	endbr64 
    78a4:	48 63 d2             	movslq %edx,%rdx
    78a7:	48 8d 05 12 99 00 00 	lea    0x9912(%rip),%rax        # 111c0 <galois_div_tables>
    78ae:	48 89 d1             	mov    %rdx,%rcx
    78b1:	48 8b 04 d0          	mov    (%rax,%rdx,8),%rax
    78b5:	d3 e7                	shl    %cl,%edi
    78b7:	09 f7                	or     %esi,%edi
    78b9:	48 63 ff             	movslq %edi,%rdi
    78bc:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    78bf:	c3                   	retq   

00000000000078c0 <galois_w08_region_multiply>:
    78c0:	f3 0f 1e fa          	endbr64 
    78c4:	41 54                	push   %r12
    78c6:	41 89 f1             	mov    %esi,%r9d
    78c9:	55                   	push   %rbp
    78ca:	89 d5                	mov    %edx,%ebp
    78cc:	53                   	push   %rbx
    78cd:	48 89 fb             	mov    %rdi,%rbx
    78d0:	48 83 ec 20          	sub    $0x20,%rsp
    78d4:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    78db:	00 00 
    78dd:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    78e2:	31 c0                	xor    %eax,%eax
    78e4:	48 8b 05 35 9a 00 00 	mov    0x9a35(%rip),%rax        # 11320 <galois_mult_tables+0x40>
    78eb:	48 85 c9             	test   %rcx,%rcx
    78ee:	0f 84 80 00 00 00    	je     7974 <galois_w08_region_multiply+0xb4>
    78f4:	49 89 cc             	mov    %rcx,%r12
    78f7:	48 85 c0             	test   %rax,%rax
    78fa:	0f 84 b6 00 00 00    	je     79b6 <galois_w08_region_multiply+0xf6>
    7900:	41 c1 e1 08          	shl    $0x8,%r9d
    7904:	45 85 c0             	test   %r8d,%r8d
    7907:	74 7b                	je     7984 <galois_w08_region_multiply+0xc4>
    7909:	85 ed                	test   %ebp,%ebp
    790b:	7e 4a                	jle    7957 <galois_w08_region_multiply+0x97>
    790d:	48 8b 15 0c 9a 00 00 	mov    0x9a0c(%rip),%rdx        # 11320 <galois_mult_tables+0x40>
    7914:	48 89 df             	mov    %rbx,%rdi
    7917:	31 c9                	xor    %ecx,%ecx
    7919:	4c 8d 44 24 10       	lea    0x10(%rsp),%r8
    791e:	66 90                	xchg   %ax,%ax
    7920:	31 f6                	xor    %esi,%esi
    7922:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7928:	0f b6 04 37          	movzbl (%rdi,%rsi,1),%eax
    792c:	44 01 c8             	add    %r9d,%eax
    792f:	48 98                	cltq   
    7931:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    7934:	41 88 04 30          	mov    %al,(%r8,%rsi,1)
    7938:	48 83 c6 01          	add    $0x1,%rsi
    793c:	48 83 fe 08          	cmp    $0x8,%rsi
    7940:	75 e6                	jne    7928 <galois_w08_region_multiply+0x68>
    7942:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    7947:	49 31 04 0c          	xor    %rax,(%r12,%rcx,1)
    794b:	48 83 c1 08          	add    $0x8,%rcx
    794f:	48 83 c7 08          	add    $0x8,%rdi
    7953:	39 cd                	cmp    %ecx,%ebp
    7955:	7f c9                	jg     7920 <galois_w08_region_multiply+0x60>
    7957:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    795c:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    7963:	00 00 
    7965:	0f 85 ba 00 00 00    	jne    7a25 <galois_w08_region_multiply+0x165>
    796b:	48 83 c4 20          	add    $0x20,%rsp
    796f:	5b                   	pop    %rbx
    7970:	5d                   	pop    %rbp
    7971:	41 5c                	pop    %r12
    7973:	c3                   	retq   
    7974:	48 85 c0             	test   %rax,%rax
    7977:	0f 84 8b 00 00 00    	je     7a08 <galois_w08_region_multiply+0x148>
    797d:	41 c1 e1 08          	shl    $0x8,%r9d
    7981:	49 89 dc             	mov    %rbx,%r12
    7984:	85 ed                	test   %ebp,%ebp
    7986:	7e cf                	jle    7957 <galois_w08_region_multiply+0x97>
    7988:	48 8b 35 91 99 00 00 	mov    0x9991(%rip),%rsi        # 11320 <galois_mult_tables+0x40>
    798f:	8d 4d ff             	lea    -0x1(%rbp),%ecx
    7992:	31 d2                	xor    %edx,%edx
    7994:	0f 1f 40 00          	nopl   0x0(%rax)
    7998:	0f b6 04 13          	movzbl (%rbx,%rdx,1),%eax
    799c:	44 01 c8             	add    %r9d,%eax
    799f:	48 98                	cltq   
    79a1:	8b 04 86             	mov    (%rsi,%rax,4),%eax
    79a4:	41 88 04 14          	mov    %al,(%r12,%rdx,1)
    79a8:	48 89 d0             	mov    %rdx,%rax
    79ab:	48 83 c2 01          	add    $0x1,%rdx
    79af:	48 39 c8             	cmp    %rcx,%rax
    79b2:	75 e4                	jne    7998 <galois_w08_region_multiply+0xd8>
    79b4:	eb a1                	jmp    7957 <galois_w08_region_multiply+0x97>
    79b6:	bf 08 00 00 00       	mov    $0x8,%edi
    79bb:	44 89 44 24 0c       	mov    %r8d,0xc(%rsp)
    79c0:	89 74 24 08          	mov    %esi,0x8(%rsp)
    79c4:	e8 a7 fa ff ff       	callq  7470 <galois_create_mult_tables>
    79c9:	44 8b 4c 24 08       	mov    0x8(%rsp),%r9d
    79ce:	44 8b 44 24 0c       	mov    0xc(%rsp),%r8d
    79d3:	85 c0                	test   %eax,%eax
    79d5:	0f 89 25 ff ff ff    	jns    7900 <galois_w08_region_multiply+0x40>
    79db:	48 8b 0d 5e 97 00 00 	mov    0x975e(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    79e2:	ba 41 00 00 00       	mov    $0x41,%edx
    79e7:	be 01 00 00 00       	mov    $0x1,%esi
    79ec:	48 8d 3d 35 2f 00 00 	lea    0x2f35(%rip),%rdi        # a928 <__PRETTY_FUNCTION__.5230+0x257>
    79f3:	e8 68 9a ff ff       	callq  1460 <fwrite@plt>
    79f8:	bf 01 00 00 00       	mov    $0x1,%edi
    79fd:	e8 4e 9a ff ff       	callq  1450 <exit@plt>
    7a02:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7a08:	bf 08 00 00 00       	mov    $0x8,%edi
    7a0d:	89 74 24 08          	mov    %esi,0x8(%rsp)
    7a11:	e8 5a fa ff ff       	callq  7470 <galois_create_mult_tables>
    7a16:	44 8b 4c 24 08       	mov    0x8(%rsp),%r9d
    7a1b:	85 c0                	test   %eax,%eax
    7a1d:	0f 89 5a ff ff ff    	jns    797d <galois_w08_region_multiply+0xbd>
    7a23:	eb b6                	jmp    79db <galois_w08_region_multiply+0x11b>
    7a25:	e8 d6 98 ff ff       	callq  1300 <__stack_chk_fail@plt>
    7a2a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000007a30 <galois_w16_region_multiply>:
    7a30:	f3 0f 1e fa          	endbr64 
    7a34:	41 55                	push   %r13
    7a36:	41 54                	push   %r12
    7a38:	55                   	push   %rbp
    7a39:	89 d5                	mov    %edx,%ebp
    7a3b:	53                   	push   %rbx
    7a3c:	48 89 cb             	mov    %rcx,%rbx
    7a3f:	48 83 ec 38          	sub    $0x38,%rsp
    7a43:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    7a4a:	00 00 
    7a4c:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    7a51:	31 c0                	xor    %eax,%eax
    7a53:	48 85 c9             	test   %rcx,%rcx
    7a56:	48 0f 44 df          	cmove  %rdi,%rbx
    7a5a:	c1 ed 1f             	shr    $0x1f,%ebp
    7a5d:	01 d5                	add    %edx,%ebp
    7a5f:	d1 fd                	sar    %ebp
    7a61:	85 f6                	test   %esi,%esi
    7a63:	0f 84 f7 00 00 00    	je     7b60 <galois_w16_region_multiply+0x130>
    7a69:	4c 8b 0d 30 9b 00 00 	mov    0x9b30(%rip),%r9        # 115a0 <galois_log_tables+0x80>
    7a70:	49 89 fd             	mov    %rdi,%r13
    7a73:	4d 85 c9             	test   %r9,%r9
    7a76:	0f 84 19 01 00 00    	je     7b95 <galois_w16_region_multiply+0x165>
    7a7c:	4c 63 e6             	movslq %esi,%r12
    7a7f:	47 8b 1c a1          	mov    (%r9,%r12,4),%r11d
    7a83:	48 85 c9             	test   %rcx,%rcx
    7a86:	74 78                	je     7b00 <galois_w16_region_multiply+0xd0>
    7a88:	45 85 c0             	test   %r8d,%r8d
    7a8b:	74 73                	je     7b00 <galois_w16_region_multiply+0xd0>
    7a8d:	83 fa 01             	cmp    $0x1,%edx
    7a90:	7e 4d                	jle    7adf <galois_w16_region_multiply+0xaf>
    7a92:	4c 8b 64 24 20       	mov    0x20(%rsp),%r12
    7a97:	4c 89 ef             	mov    %r13,%rdi
    7a9a:	31 c9                	xor    %ecx,%ecx
    7a9c:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
    7aa1:	4c 8b 05 d8 99 00 00 	mov    0x99d8(%rip),%r8        # 11480 <galois_ilog_tables+0x80>
    7aa8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    7aaf:	00 
    7ab0:	31 c0                	xor    %eax,%eax
    7ab2:	0f b7 14 07          	movzwl (%rdi,%rax,1),%edx
    7ab6:	66 85 d2             	test   %dx,%dx
    7ab9:	0f 85 81 00 00 00    	jne    7b40 <galois_w16_region_multiply+0x110>
    7abf:	31 d2                	xor    %edx,%edx
    7ac1:	66 89 14 06          	mov    %dx,(%rsi,%rax,1)
    7ac5:	48 83 c0 02          	add    $0x2,%rax
    7ac9:	48 83 f8 08          	cmp    $0x8,%rax
    7acd:	75 e3                	jne    7ab2 <galois_w16_region_multiply+0x82>
    7acf:	4c 31 24 4b          	xor    %r12,(%rbx,%rcx,2)
    7ad3:	48 83 c1 04          	add    $0x4,%rcx
    7ad7:	48 83 c7 08          	add    $0x8,%rdi
    7adb:	39 cd                	cmp    %ecx,%ebp
    7add:	7f d1                	jg     7ab0 <galois_w16_region_multiply+0x80>
    7adf:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    7ae4:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    7aeb:	00 00 
    7aed:	0f 85 07 01 00 00    	jne    7bfa <galois_w16_region_multiply+0x1ca>
    7af3:	48 83 c4 38          	add    $0x38,%rsp
    7af7:	5b                   	pop    %rbx
    7af8:	5d                   	pop    %rbp
    7af9:	41 5c                	pop    %r12
    7afb:	41 5d                	pop    %r13
    7afd:	c3                   	retq   
    7afe:	66 90                	xchg   %ax,%ax
    7b00:	83 fa 01             	cmp    $0x1,%edx
    7b03:	7e da                	jle    7adf <galois_w16_region_multiply+0xaf>
    7b05:	48 8b 0d 74 99 00 00 	mov    0x9974(%rip),%rcx        # 11480 <galois_ilog_tables+0x80>
    7b0c:	31 c0                	xor    %eax,%eax
    7b0e:	eb 0e                	jmp    7b1e <galois_w16_region_multiply+0xee>
    7b10:	31 f6                	xor    %esi,%esi
    7b12:	66 89 34 43          	mov    %si,(%rbx,%rax,2)
    7b16:	48 83 c0 01          	add    $0x1,%rax
    7b1a:	39 c5                	cmp    %eax,%ebp
    7b1c:	7e c1                	jle    7adf <galois_w16_region_multiply+0xaf>
    7b1e:	41 0f b7 54 45 00    	movzwl 0x0(%r13,%rax,2),%edx
    7b24:	66 85 d2             	test   %dx,%dx
    7b27:	74 e7                	je     7b10 <galois_w16_region_multiply+0xe0>
    7b29:	41 8b 3c 91          	mov    (%r9,%rdx,4),%edi
    7b2d:	44 01 df             	add    %r11d,%edi
    7b30:	48 63 d7             	movslq %edi,%rdx
    7b33:	8b 14 91             	mov    (%rcx,%rdx,4),%edx
    7b36:	66 89 14 43          	mov    %dx,(%rbx,%rax,2)
    7b3a:	eb da                	jmp    7b16 <galois_w16_region_multiply+0xe6>
    7b3c:	0f 1f 40 00          	nopl   0x0(%rax)
    7b40:	45 8b 2c 91          	mov    (%r9,%rdx,4),%r13d
    7b44:	45 01 dd             	add    %r11d,%r13d
    7b47:	49 63 d5             	movslq %r13d,%rdx
    7b4a:	41 8b 14 90          	mov    (%r8,%rdx,4),%edx
    7b4e:	66 89 14 06          	mov    %dx,(%rsi,%rax,1)
    7b52:	e9 6e ff ff ff       	jmpq   7ac5 <galois_w16_region_multiply+0x95>
    7b57:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    7b5e:	00 00 
    7b60:	45 85 c0             	test   %r8d,%r8d
    7b63:	0f 85 76 ff ff ff    	jne    7adf <galois_w16_region_multiply+0xaf>
    7b69:	48 63 ed             	movslq %ebp,%rbp
    7b6c:	48 8d 04 6b          	lea    (%rbx,%rbp,2),%rax
    7b70:	48 39 c3             	cmp    %rax,%rbx
    7b73:	0f 83 66 ff ff ff    	jae    7adf <galois_w16_region_multiply+0xaf>
    7b79:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7b80:	48 c7 03 00 00 00 00 	movq   $0x0,(%rbx)
    7b87:	48 83 c3 08          	add    $0x8,%rbx
    7b8b:	48 39 d8             	cmp    %rbx,%rax
    7b8e:	77 f0                	ja     7b80 <galois_w16_region_multiply+0x150>
    7b90:	e9 4a ff ff ff       	jmpq   7adf <galois_w16_region_multiply+0xaf>
    7b95:	bf 10 00 00 00       	mov    $0x10,%edi
    7b9a:	44 89 44 24 1c       	mov    %r8d,0x1c(%rsp)
    7b9f:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    7ba4:	89 54 24 18          	mov    %edx,0x18(%rsp)
    7ba8:	89 74 24 0c          	mov    %esi,0xc(%rsp)
    7bac:	e8 5f f6 ff ff       	callq  7210 <galois_create_log_tables.part.0>
    7bb1:	85 c0                	test   %eax,%eax
    7bb3:	78 1e                	js     7bd3 <galois_w16_region_multiply+0x1a3>
    7bb5:	4c 8b 0d e4 99 00 00 	mov    0x99e4(%rip),%r9        # 115a0 <galois_log_tables+0x80>
    7bbc:	8b 74 24 0c          	mov    0xc(%rsp),%esi
    7bc0:	8b 54 24 18          	mov    0x18(%rsp),%edx
    7bc4:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    7bc9:	44 8b 44 24 1c       	mov    0x1c(%rsp),%r8d
    7bce:	e9 a9 fe ff ff       	jmpq   7a7c <galois_w16_region_multiply+0x4c>
    7bd3:	48 8b 0d 66 95 00 00 	mov    0x9566(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    7bda:	ba 36 00 00 00       	mov    $0x36,%edx
    7bdf:	be 01 00 00 00       	mov    $0x1,%esi
    7be4:	48 8d 3d 85 2d 00 00 	lea    0x2d85(%rip),%rdi        # a970 <__PRETTY_FUNCTION__.5230+0x29f>
    7beb:	e8 70 98 ff ff       	callq  1460 <fwrite@plt>
    7bf0:	bf 01 00 00 00       	mov    $0x1,%edi
    7bf5:	e8 56 98 ff ff       	callq  1450 <exit@plt>
    7bfa:	e8 01 97 ff ff       	callq  1300 <__stack_chk_fail@plt>
    7bff:	90                   	nop

0000000000007c00 <galois_invert_binary_matrix>:
    7c00:	f3 0f 1e fa          	endbr64 
    7c04:	85 d2                	test   %edx,%edx
    7c06:	0f 8e 4a 01 00 00    	jle    7d56 <galois_invert_binary_matrix+0x156>
    7c0c:	41 56                	push   %r14
    7c0e:	41 89 d1             	mov    %edx,%r9d
    7c11:	31 c9                	xor    %ecx,%ecx
    7c13:	41 55                	push   %r13
    7c15:	41 54                	push   %r12
    7c17:	55                   	push   %rbp
    7c18:	53                   	push   %rbx
    7c19:	8d 5a ff             	lea    -0x1(%rdx),%ebx
    7c1c:	ba 01 00 00 00       	mov    $0x1,%edx
    7c21:	49 89 db             	mov    %rbx,%r11
    7c24:	4c 8d 63 01          	lea    0x1(%rbx),%r12
    7c28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    7c2f:	00 
    7c30:	89 d0                	mov    %edx,%eax
    7c32:	d3 e0                	shl    %cl,%eax
    7c34:	89 04 8e             	mov    %eax,(%rsi,%rcx,4)
    7c37:	48 89 c8             	mov    %rcx,%rax
    7c3a:	48 83 c1 01          	add    $0x1,%rcx
    7c3e:	48 39 d8             	cmp    %rbx,%rax
    7c41:	75 ed                	jne    7c30 <galois_invert_binary_matrix+0x30>
    7c43:	48 83 c3 02          	add    $0x2,%rbx
    7c47:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    7c4d:	bd 01 00 00 00       	mov    $0x1,%ebp
    7c52:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7c58:	46 8b 6c 87 fc       	mov    -0x4(%rdi,%r8,4),%r13d
    7c5d:	41 8d 48 ff          	lea    -0x1(%r8),%ecx
    7c61:	44 89 c2             	mov    %r8d,%edx
    7c64:	41 0f a3 cd          	bt     %ecx,%r13d
    7c68:	0f 83 92 00 00 00    	jae    7d00 <galois_invert_binary_matrix+0x100>
    7c6e:	4d 39 e0             	cmp    %r12,%r8
    7c71:	74 3d                	je     7cb0 <galois_invert_binary_matrix+0xb0>
    7c73:	89 e8                	mov    %ebp,%eax
    7c75:	d3 e0                	shl    %cl,%eax
    7c77:	89 c1                	mov    %eax,%ecx
    7c79:	4c 89 c0             	mov    %r8,%rax
    7c7c:	0f 1f 40 00          	nopl   0x0(%rax)
    7c80:	8b 14 87             	mov    (%rdi,%rax,4),%edx
    7c83:	85 ca                	test   %ecx,%edx
    7c85:	74 10                	je     7c97 <galois_invert_binary_matrix+0x97>
    7c87:	42 33 54 87 fc       	xor    -0x4(%rdi,%r8,4),%edx
    7c8c:	89 14 87             	mov    %edx,(%rdi,%rax,4)
    7c8f:	42 8b 54 86 fc       	mov    -0x4(%rsi,%r8,4),%edx
    7c94:	31 14 86             	xor    %edx,(%rsi,%rax,4)
    7c97:	48 83 c0 01          	add    $0x1,%rax
    7c9b:	41 39 c1             	cmp    %eax,%r9d
    7c9e:	75 e0                	jne    7c80 <galois_invert_binary_matrix+0x80>
    7ca0:	49 83 c0 01          	add    $0x1,%r8
    7ca4:	4c 39 c3             	cmp    %r8,%rbx
    7ca7:	75 af                	jne    7c58 <galois_invert_binary_matrix+0x58>
    7ca9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7cb0:	49 63 cb             	movslq %r11d,%rcx
    7cb3:	41 bb 01 00 00 00    	mov    $0x1,%r11d
    7cb9:	85 c9                	test   %ecx,%ecx
    7cbb:	7e 33                	jle    7cf0 <galois_invert_binary_matrix+0xf0>
    7cbd:	45 89 d9             	mov    %r11d,%r9d
    7cc0:	44 8d 41 ff          	lea    -0x1(%rcx),%r8d
    7cc4:	31 c0                	xor    %eax,%eax
    7cc6:	41 d3 e1             	shl    %cl,%r9d
    7cc9:	eb 08                	jmp    7cd3 <galois_invert_binary_matrix+0xd3>
    7ccb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7cd0:	48 89 d0             	mov    %rdx,%rax
    7cd3:	44 85 0c 87          	test   %r9d,(%rdi,%rax,4)
    7cd7:	74 06                	je     7cdf <galois_invert_binary_matrix+0xdf>
    7cd9:	8b 14 8e             	mov    (%rsi,%rcx,4),%edx
    7cdc:	31 14 86             	xor    %edx,(%rsi,%rax,4)
    7cdf:	48 8d 50 01          	lea    0x1(%rax),%rdx
    7ce3:	4c 39 c0             	cmp    %r8,%rax
    7ce6:	75 e8                	jne    7cd0 <galois_invert_binary_matrix+0xd0>
    7ce8:	48 83 e9 01          	sub    $0x1,%rcx
    7cec:	85 c9                	test   %ecx,%ecx
    7cee:	7f cd                	jg     7cbd <galois_invert_binary_matrix+0xbd>
    7cf0:	89 c8                	mov    %ecx,%eax
    7cf2:	83 e8 01             	sub    $0x1,%eax
    7cf5:	79 f1                	jns    7ce8 <galois_invert_binary_matrix+0xe8>
    7cf7:	5b                   	pop    %rbx
    7cf8:	5d                   	pop    %rbp
    7cf9:	41 5c                	pop    %r12
    7cfb:	41 5d                	pop    %r13
    7cfd:	41 5e                	pop    %r14
    7cff:	c3                   	retq   
    7d00:	45 39 c1             	cmp    %r8d,%r9d
    7d03:	7e 1f                	jle    7d24 <galois_invert_binary_matrix+0x124>
    7d05:	41 89 ee             	mov    %ebp,%r14d
    7d08:	4c 89 c0             	mov    %r8,%rax
    7d0b:	41 d3 e6             	shl    %cl,%r14d
    7d0e:	eb 0c                	jmp    7d1c <galois_invert_binary_matrix+0x11c>
    7d10:	8d 50 01             	lea    0x1(%rax),%edx
    7d13:	48 83 c0 01          	add    $0x1,%rax
    7d17:	41 39 c1             	cmp    %eax,%r9d
    7d1a:	7e 08                	jle    7d24 <galois_invert_binary_matrix+0x124>
    7d1c:	89 c2                	mov    %eax,%edx
    7d1e:	44 85 34 87          	test   %r14d,(%rdi,%rax,4)
    7d22:	74 ec                	je     7d10 <galois_invert_binary_matrix+0x110>
    7d24:	41 39 d1             	cmp    %edx,%r9d
    7d27:	74 2e                	je     7d57 <galois_invert_binary_matrix+0x157>
    7d29:	48 63 c2             	movslq %edx,%rax
    7d2c:	48 c1 e0 02          	shl    $0x2,%rax
    7d30:	48 8d 14 07          	lea    (%rdi,%rax,1),%rdx
    7d34:	48 01 f0             	add    %rsi,%rax
    7d37:	44 8b 32             	mov    (%rdx),%r14d
    7d3a:	46 89 74 87 fc       	mov    %r14d,-0x4(%rdi,%r8,4)
    7d3f:	44 89 2a             	mov    %r13d,(%rdx)
    7d42:	42 8b 54 86 fc       	mov    -0x4(%rsi,%r8,4),%edx
    7d47:	44 8b 28             	mov    (%rax),%r13d
    7d4a:	46 89 6c 86 fc       	mov    %r13d,-0x4(%rsi,%r8,4)
    7d4f:	89 10                	mov    %edx,(%rax)
    7d51:	e9 18 ff ff ff       	jmpq   7c6e <galois_invert_binary_matrix+0x6e>
    7d56:	c3                   	retq   
    7d57:	48 8b 0d e2 93 00 00 	mov    0x93e2(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    7d5e:	ba 2e 00 00 00       	mov    $0x2e,%edx
    7d63:	be 01 00 00 00       	mov    $0x1,%esi
    7d68:	48 8d 3d 39 2c 00 00 	lea    0x2c39(%rip),%rdi        # a9a8 <__PRETTY_FUNCTION__.5230+0x2d7>
    7d6f:	e8 ec 96 ff ff       	callq  1460 <fwrite@plt>
    7d74:	bf 01 00 00 00       	mov    $0x1,%edi
    7d79:	e8 d2 96 ff ff       	callq  1450 <exit@plt>
    7d7e:	66 90                	xchg   %ax,%ax

0000000000007d80 <galois_shift_inverse>:
    7d80:	f3 0f 1e fa          	endbr64 
    7d84:	55                   	push   %rbp
    7d85:	89 f2                	mov    %esi,%edx
    7d87:	53                   	push   %rbx
    7d88:	48 81 ec 18 01 00 00 	sub    $0x118,%rsp
    7d8f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    7d96:	00 00 
    7d98:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
    7d9f:	00 
    7da0:	31 c0                	xor    %eax,%eax
    7da2:	85 f6                	test   %esi,%esi
    7da4:	0f 8e 8e 00 00 00    	jle    7e38 <galois_shift_inverse+0xb8>
    7daa:	8d 4e ff             	lea    -0x1(%rsi),%ecx
    7dad:	48 8d 05 ec 2d 00 00 	lea    0x2dec(%rip),%rax        # aba0 <nw>
    7db4:	48 89 e5             	mov    %rsp,%rbp
    7db7:	4c 63 c2             	movslq %edx,%r8
    7dba:	48 63 f1             	movslq %ecx,%rsi
    7dbd:	89 c9                	mov    %ecx,%ecx
    7dbf:	48 8d 1d 1a 2f 00 00 	lea    0x2f1a(%rip),%rbx        # ace0 <prim_poly>
    7dc6:	44 8b 1c b0          	mov    (%rax,%rsi,4),%r11d
    7dca:	4c 8d 4c 8c 04       	lea    0x4(%rsp,%rcx,4),%r9
    7dcf:	48 89 e8             	mov    %rbp,%rax
    7dd2:	48 8d 35 27 2d 00 00 	lea    0x2d27(%rip),%rsi        # ab00 <nwm1>
    7dd9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7de0:	44 89 d9             	mov    %r11d,%ecx
    7de3:	89 38                	mov    %edi,(%rax)
    7de5:	21 f9                	and    %edi,%ecx
    7de7:	01 ff                	add    %edi,%edi
    7de9:	85 c9                	test   %ecx,%ecx
    7deb:	74 08                	je     7df5 <galois_shift_inverse+0x75>
    7ded:	42 33 3c 83          	xor    (%rbx,%r8,4),%edi
    7df1:	42 23 3c 86          	and    (%rsi,%r8,4),%edi
    7df5:	48 83 c0 04          	add    $0x4,%rax
    7df9:	4c 39 c8             	cmp    %r9,%rax
    7dfc:	75 e2                	jne    7de0 <galois_shift_inverse+0x60>
    7dfe:	48 8d b4 24 80 00 00 	lea    0x80(%rsp),%rsi
    7e05:	00 
    7e06:	48 89 ef             	mov    %rbp,%rdi
    7e09:	e8 f2 fd ff ff       	callq  7c00 <galois_invert_binary_matrix>
    7e0e:	8b 84 24 80 00 00 00 	mov    0x80(%rsp),%eax
    7e15:	48 8b 9c 24 08 01 00 	mov    0x108(%rsp),%rbx
    7e1c:	00 
    7e1d:	64 48 33 1c 25 28 00 	xor    %fs:0x28,%rbx
    7e24:	00 00 
    7e26:	75 15                	jne    7e3d <galois_shift_inverse+0xbd>
    7e28:	48 81 c4 18 01 00 00 	add    $0x118,%rsp
    7e2f:	5b                   	pop    %rbx
    7e30:	5d                   	pop    %rbp
    7e31:	c3                   	retq   
    7e32:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    7e38:	48 89 e5             	mov    %rsp,%rbp
    7e3b:	eb c1                	jmp    7dfe <galois_shift_inverse+0x7e>
    7e3d:	e8 be 94 ff ff       	callq  1300 <__stack_chk_fail@plt>
    7e42:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    7e49:	00 00 00 00 
    7e4d:	0f 1f 00             	nopl   (%rax)

0000000000007e50 <galois_shift_divide>:
    7e50:	f3 0f 1e fa          	endbr64 
    7e54:	41 55                	push   %r13
    7e56:	41 54                	push   %r12
    7e58:	48 83 ec 08          	sub    $0x8,%rsp
    7e5c:	85 f6                	test   %esi,%esi
    7e5e:	74 40                	je     7ea0 <galois_shift_divide+0x50>
    7e60:	41 89 fc             	mov    %edi,%r12d
    7e63:	85 ff                	test   %edi,%edi
    7e65:	75 11                	jne    7e78 <galois_shift_divide+0x28>
    7e67:	48 83 c4 08          	add    $0x8,%rsp
    7e6b:	44 89 e0             	mov    %r12d,%eax
    7e6e:	41 5c                	pop    %r12
    7e70:	41 5d                	pop    %r13
    7e72:	c3                   	retq   
    7e73:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7e78:	89 f7                	mov    %esi,%edi
    7e7a:	89 d6                	mov    %edx,%esi
    7e7c:	41 89 d5             	mov    %edx,%r13d
    7e7f:	e8 fc fe ff ff       	callq  7d80 <galois_shift_inverse>
    7e84:	48 83 c4 08          	add    $0x8,%rsp
    7e88:	44 89 ea             	mov    %r13d,%edx
    7e8b:	44 89 e7             	mov    %r12d,%edi
    7e8e:	89 c6                	mov    %eax,%esi
    7e90:	41 5c                	pop    %r12
    7e92:	41 5d                	pop    %r13
    7e94:	e9 d7 f8 ff ff       	jmpq   7770 <galois_shift_multiply>
    7e99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7ea0:	41 bc ff ff ff ff    	mov    $0xffffffff,%r12d
    7ea6:	eb bf                	jmp    7e67 <galois_shift_divide+0x17>
    7ea8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    7eaf:	00 

0000000000007eb0 <galois_get_mult_table>:
    7eb0:	f3 0f 1e fa          	endbr64 
    7eb4:	41 54                	push   %r12
    7eb6:	55                   	push   %rbp
    7eb7:	48 63 ef             	movslq %edi,%rbp
    7eba:	53                   	push   %rbx
    7ebb:	48 8d 1d 1e 94 00 00 	lea    0x941e(%rip),%rbx        # 112e0 <galois_mult_tables>
    7ec2:	4c 8b 24 eb          	mov    (%rbx,%rbp,8),%r12
    7ec6:	4d 85 e4             	test   %r12,%r12
    7ec9:	74 0d                	je     7ed8 <galois_get_mult_table+0x28>
    7ecb:	4c 89 e0             	mov    %r12,%rax
    7ece:	5b                   	pop    %rbx
    7ecf:	5d                   	pop    %rbp
    7ed0:	41 5c                	pop    %r12
    7ed2:	c3                   	retq   
    7ed3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7ed8:	e8 93 f5 ff ff       	callq  7470 <galois_create_mult_tables>
    7edd:	85 c0                	test   %eax,%eax
    7edf:	75 ea                	jne    7ecb <galois_get_mult_table+0x1b>
    7ee1:	4c 8b 24 eb          	mov    (%rbx,%rbp,8),%r12
    7ee5:	5b                   	pop    %rbx
    7ee6:	5d                   	pop    %rbp
    7ee7:	4c 89 e0             	mov    %r12,%rax
    7eea:	41 5c                	pop    %r12
    7eec:	c3                   	retq   
    7eed:	0f 1f 00             	nopl   (%rax)

0000000000007ef0 <galois_get_div_table>:
    7ef0:	f3 0f 1e fa          	endbr64 
    7ef4:	41 54                	push   %r12
    7ef6:	48 8d 05 e3 93 00 00 	lea    0x93e3(%rip),%rax        # 112e0 <galois_mult_tables>
    7efd:	53                   	push   %rbx
    7efe:	48 63 df             	movslq %edi,%rbx
    7f01:	48 83 ec 08          	sub    $0x8,%rsp
    7f05:	4c 8b 24 d8          	mov    (%rax,%rbx,8),%r12
    7f09:	4d 85 e4             	test   %r12,%r12
    7f0c:	74 1a                	je     7f28 <galois_get_div_table+0x38>
    7f0e:	48 8d 05 ab 92 00 00 	lea    0x92ab(%rip),%rax        # 111c0 <galois_div_tables>
    7f15:	4c 8b 24 d8          	mov    (%rax,%rbx,8),%r12
    7f19:	48 83 c4 08          	add    $0x8,%rsp
    7f1d:	4c 89 e0             	mov    %r12,%rax
    7f20:	5b                   	pop    %rbx
    7f21:	41 5c                	pop    %r12
    7f23:	c3                   	retq   
    7f24:	0f 1f 40 00          	nopl   0x0(%rax)
    7f28:	e8 43 f5 ff ff       	callq  7470 <galois_create_mult_tables>
    7f2d:	85 c0                	test   %eax,%eax
    7f2f:	74 dd                	je     7f0e <galois_get_div_table+0x1e>
    7f31:	eb e6                	jmp    7f19 <galois_get_div_table+0x29>
    7f33:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    7f3a:	00 00 00 00 
    7f3e:	66 90                	xchg   %ax,%ax

0000000000007f40 <galois_get_log_table>:
    7f40:	f3 0f 1e fa          	endbr64 
    7f44:	41 54                	push   %r12
    7f46:	55                   	push   %rbp
    7f47:	48 63 ef             	movslq %edi,%rbp
    7f4a:	53                   	push   %rbx
    7f4b:	48 8d 1d ce 95 00 00 	lea    0x95ce(%rip),%rbx        # 11520 <galois_log_tables>
    7f52:	4c 8b 24 eb          	mov    (%rbx,%rbp,8),%r12
    7f56:	4d 85 e4             	test   %r12,%r12
    7f59:	74 0d                	je     7f68 <galois_get_log_table+0x28>
    7f5b:	4c 89 e0             	mov    %r12,%rax
    7f5e:	5b                   	pop    %rbx
    7f5f:	5d                   	pop    %rbp
    7f60:	41 5c                	pop    %r12
    7f62:	c3                   	retq   
    7f63:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7f68:	83 ff 1e             	cmp    $0x1e,%edi
    7f6b:	7f ee                	jg     7f5b <galois_get_log_table+0x1b>
    7f6d:	e8 9e f2 ff ff       	callq  7210 <galois_create_log_tables.part.0>
    7f72:	85 c0                	test   %eax,%eax
    7f74:	75 e5                	jne    7f5b <galois_get_log_table+0x1b>
    7f76:	4c 8b 24 eb          	mov    (%rbx,%rbp,8),%r12
    7f7a:	5b                   	pop    %rbx
    7f7b:	5d                   	pop    %rbp
    7f7c:	4c 89 e0             	mov    %r12,%rax
    7f7f:	41 5c                	pop    %r12
    7f81:	c3                   	retq   
    7f82:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    7f89:	00 00 00 00 
    7f8d:	0f 1f 00             	nopl   (%rax)

0000000000007f90 <galois_get_ilog_table>:
    7f90:	f3 0f 1e fa          	endbr64 
    7f94:	41 54                	push   %r12
    7f96:	55                   	push   %rbp
    7f97:	48 8d 2d 62 94 00 00 	lea    0x9462(%rip),%rbp        # 11400 <galois_ilog_tables>
    7f9e:	53                   	push   %rbx
    7f9f:	48 63 df             	movslq %edi,%rbx
    7fa2:	4c 8b 64 dd 00       	mov    0x0(%rbp,%rbx,8),%r12
    7fa7:	4d 85 e4             	test   %r12,%r12
    7faa:	74 0c                	je     7fb8 <galois_get_ilog_table+0x28>
    7fac:	4c 89 e0             	mov    %r12,%rax
    7faf:	5b                   	pop    %rbx
    7fb0:	5d                   	pop    %rbp
    7fb1:	41 5c                	pop    %r12
    7fb3:	c3                   	retq   
    7fb4:	0f 1f 40 00          	nopl   0x0(%rax)
    7fb8:	83 ff 1e             	cmp    $0x1e,%edi
    7fbb:	7f ef                	jg     7fac <galois_get_ilog_table+0x1c>
    7fbd:	48 8d 05 5c 95 00 00 	lea    0x955c(%rip),%rax        # 11520 <galois_log_tables>
    7fc4:	48 83 3c d8 00       	cmpq   $0x0,(%rax,%rbx,8)
    7fc9:	75 e1                	jne    7fac <galois_get_ilog_table+0x1c>
    7fcb:	e8 40 f2 ff ff       	callq  7210 <galois_create_log_tables.part.0>
    7fd0:	85 c0                	test   %eax,%eax
    7fd2:	75 d8                	jne    7fac <galois_get_ilog_table+0x1c>
    7fd4:	4c 8b 64 dd 00       	mov    0x0(%rbp,%rbx,8),%r12
    7fd9:	eb d1                	jmp    7fac <galois_get_ilog_table+0x1c>
    7fdb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000007fe0 <galois_region_xor>:
    7fe0:	f3 0f 1e fa          	endbr64 
    7fe4:	4c 63 c1             	movslq %ecx,%r8
    7fe7:	49 01 f8             	add    %rdi,%r8
    7fea:	4c 39 c7             	cmp    %r8,%rdi
    7fed:	73 29                	jae    8018 <galois_region_xor+0x38>
    7fef:	48 89 f8             	mov    %rdi,%rax
    7ff2:	48 f7 d0             	not    %rax
    7ff5:	49 01 c0             	add    %rax,%r8
    7ff8:	31 c0                	xor    %eax,%eax
    7ffa:	49 c1 e8 03          	shr    $0x3,%r8
    7ffe:	66 90                	xchg   %ax,%ax
    8000:	48 8b 0c c7          	mov    (%rdi,%rax,8),%rcx
    8004:	48 33 0c c6          	xor    (%rsi,%rax,8),%rcx
    8008:	48 89 0c c2          	mov    %rcx,(%rdx,%rax,8)
    800c:	48 89 c1             	mov    %rax,%rcx
    800f:	48 83 c0 01          	add    $0x1,%rax
    8013:	49 39 c8             	cmp    %rcx,%r8
    8016:	75 e8                	jne    8000 <galois_region_xor+0x20>
    8018:	c3                   	retq   
    8019:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000008020 <galois_create_split_w8_tables>:
    8020:	f3 0f 1e fa          	endbr64 
    8024:	48 83 3d 54 91 00 00 	cmpq   $0x0,0x9154(%rip)        # 11180 <galois_split_w8>
    802b:	00 
    802c:	74 03                	je     8031 <galois_create_split_w8_tables+0x11>
    802e:	31 c0                	xor    %eax,%eax
    8030:	c3                   	retq   
    8031:	41 57                	push   %r15
    8033:	bf 08 00 00 00       	mov    $0x8,%edi
    8038:	41 56                	push   %r14
    803a:	41 55                	push   %r13
    803c:	41 54                	push   %r12
    803e:	55                   	push   %rbp
    803f:	53                   	push   %rbx
    8040:	48 83 ec 18          	sub    $0x18,%rsp
    8044:	e8 27 f4 ff ff       	callq  7470 <galois_create_mult_tables>
    8049:	85 c0                	test   %eax,%eax
    804b:	0f 88 0e 01 00 00    	js     815f <galois_create_split_w8_tables+0x13f>
    8051:	31 db                	xor    %ebx,%ebx
    8053:	bf 00 00 04 00       	mov    $0x40000,%edi
    8058:	89 dd                	mov    %ebx,%ebp
    805a:	e8 91 93 ff ff       	callq  13f0 <malloc@plt>
    805f:	48 8d 15 1a 91 00 00 	lea    0x911a(%rip),%rdx        # 11180 <galois_split_w8>
    8066:	48 89 04 da          	mov    %rax,(%rdx,%rbx,8)
    806a:	48 85 c0             	test   %rax,%rax
    806d:	0f 84 c9 00 00 00    	je     813c <galois_create_split_w8_tables+0x11c>
    8073:	48 83 c3 01          	add    $0x1,%rbx
    8077:	48 83 fb 07          	cmp    $0x7,%rbx
    807b:	75 d6                	jne    8053 <galois_create_split_w8_tables+0x33>
    807d:	c7 44 24 04 00 00 00 	movl   $0x0,0x4(%rsp)
    8084:	00 
    8085:	8b 44 24 04          	mov    0x4(%rsp),%eax
    8089:	45 31 e4             	xor    %r12d,%r12d
    808c:	48 8d 15 ed 90 00 00 	lea    0x90ed(%rip),%rdx        # 11180 <galois_split_w8>
    8093:	85 c0                	test   %eax,%eax
    8095:	44 8d 34 c5 00 00 00 	lea    0x0(,%rax,8),%r14d
    809c:	00 
    809d:	41 0f 95 c4          	setne  %r12b
    80a1:	44 01 e0             	add    %r12d,%eax
    80a4:	41 c1 e4 03          	shl    $0x3,%r12d
    80a8:	48 98                	cltq   
    80aa:	48 8d 04 c2          	lea    (%rdx,%rax,8),%rax
    80ae:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    80b3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    80b8:	45 31 ed             	xor    %r13d,%r13d
    80bb:	48 8b 18             	mov    (%rax),%rbx
    80be:	44 89 ed             	mov    %r13d,%ebp
    80c1:	44 89 f1             	mov    %r14d,%ecx
    80c4:	45 31 ff             	xor    %r15d,%r15d
    80c7:	d3 e5                	shl    %cl,%ebp
    80c9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    80d0:	44 89 fe             	mov    %r15d,%esi
    80d3:	44 89 e1             	mov    %r12d,%ecx
    80d6:	ba 20 00 00 00       	mov    $0x20,%edx
    80db:	89 ef                	mov    %ebp,%edi
    80dd:	d3 e6                	shl    %cl,%esi
    80df:	e8 8c f6 ff ff       	callq  7770 <galois_shift_multiply>
    80e4:	42 89 04 bb          	mov    %eax,(%rbx,%r15,4)
    80e8:	49 83 c7 01          	add    $0x1,%r15
    80ec:	49 81 ff 00 01 00 00 	cmp    $0x100,%r15
    80f3:	75 db                	jne    80d0 <galois_create_split_w8_tables+0xb0>
    80f5:	41 83 c5 01          	add    $0x1,%r13d
    80f9:	48 81 c3 00 04 00 00 	add    $0x400,%rbx
    8100:	41 81 fd 00 01 00 00 	cmp    $0x100,%r13d
    8107:	75 b5                	jne    80be <galois_create_split_w8_tables+0x9e>
    8109:	41 83 c4 08          	add    $0x8,%r12d
    810d:	48 83 44 24 08 08    	addq   $0x8,0x8(%rsp)
    8113:	41 83 fc 20          	cmp    $0x20,%r12d
    8117:	75 9a                	jne    80b3 <galois_create_split_w8_tables+0x93>
    8119:	83 44 24 04 03       	addl   $0x3,0x4(%rsp)
    811e:	8b 44 24 04          	mov    0x4(%rsp),%eax
    8122:	83 f8 06             	cmp    $0x6,%eax
    8125:	0f 85 5a ff ff ff    	jne    8085 <galois_create_split_w8_tables+0x65>
    812b:	31 c0                	xor    %eax,%eax
    812d:	48 83 c4 18          	add    $0x18,%rsp
    8131:	5b                   	pop    %rbx
    8132:	5d                   	pop    %rbp
    8133:	41 5c                	pop    %r12
    8135:	41 5d                	pop    %r13
    8137:	41 5e                	pop    %r14
    8139:	41 5f                	pop    %r15
    813b:	c3                   	retq   
    813c:	8d 5b ff             	lea    -0x1(%rbx),%ebx
    813f:	85 ed                	test   %ebp,%ebp
    8141:	74 1c                	je     815f <galois_create_split_w8_tables+0x13f>
    8143:	48 63 db             	movslq %ebx,%rbx
    8146:	48 8d 05 33 90 00 00 	lea    0x9033(%rip),%rax        # 11180 <galois_split_w8>
    814d:	48 8b 3c d8          	mov    (%rax,%rbx,8),%rdi
    8151:	48 83 eb 01          	sub    $0x1,%rbx
    8155:	e8 16 91 ff ff       	callq  1270 <free@plt>
    815a:	83 fb ff             	cmp    $0xffffffff,%ebx
    815d:	75 e7                	jne    8146 <galois_create_split_w8_tables+0x126>
    815f:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    8164:	eb c7                	jmp    812d <galois_create_split_w8_tables+0x10d>
    8166:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    816d:	00 00 00 

0000000000008170 <galois_w32_region_multiply>:
    8170:	f3 0f 1e fa          	endbr64 
    8174:	41 57                	push   %r15
    8176:	4c 63 fa             	movslq %edx,%r15
    8179:	41 56                	push   %r14
    817b:	45 89 c6             	mov    %r8d,%r14d
    817e:	41 55                	push   %r13
    8180:	41 89 f5             	mov    %esi,%r13d
    8183:	41 54                	push   %r12
    8185:	55                   	push   %rbp
    8186:	48 89 cd             	mov    %rcx,%rbp
    8189:	53                   	push   %rbx
    818a:	48 89 fb             	mov    %rdi,%rbx
    818d:	48 83 ec 28          	sub    $0x28,%rsp
    8191:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    8198:	00 00 
    819a:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    819f:	31 c0                	xor    %eax,%eax
    81a1:	48 85 c9             	test   %rcx,%rcx
    81a4:	48 0f 44 ef          	cmove  %rdi,%rbp
    81a8:	49 c1 ef 02          	shr    $0x2,%r15
    81ac:	48 83 3d cc 8f 00 00 	cmpq   $0x0,0x8fcc(%rip)        # 11180 <galois_split_w8>
    81b3:	00 
    81b4:	0f 84 25 01 00 00    	je     82df <galois_w32_region_multiply+0x16f>
    81ba:	49 89 e4             	mov    %rsp,%r12
    81bd:	31 c9                	xor    %ecx,%ecx
    81bf:	4c 89 e2             	mov    %r12,%rdx
    81c2:	44 89 e8             	mov    %r13d,%eax
    81c5:	48 83 c2 04          	add    $0x4,%rdx
    81c9:	d3 f8                	sar    %cl,%eax
    81cb:	83 c1 08             	add    $0x8,%ecx
    81ce:	c1 e0 08             	shl    $0x8,%eax
    81d1:	25 ff ff 00 00       	and    $0xffff,%eax
    81d6:	89 42 fc             	mov    %eax,-0x4(%rdx)
    81d9:	83 f9 20             	cmp    $0x20,%ecx
    81dc:	75 e4                	jne    81c2 <galois_w32_region_multiply+0x52>
    81de:	45 85 f6             	test   %r14d,%r14d
    81e1:	75 6e                	jne    8251 <galois_w32_region_multiply+0xe1>
    81e3:	45 85 ff             	test   %r15d,%r15d
    81e6:	0f 8e d4 00 00 00    	jle    82c0 <galois_w32_region_multiply+0x150>
    81ec:	45 8d 77 ff          	lea    -0x1(%r15),%r14d
    81f0:	45 31 ed             	xor    %r13d,%r13d
    81f3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    81f8:	46 8b 0c ab          	mov    (%rbx,%r13,4),%r9d
    81fc:	4c 8d 1d 7d 8f 00 00 	lea    0x8f7d(%rip),%r11        # 11180 <galois_split_w8>
    8203:	31 f6                	xor    %esi,%esi
    8205:	45 31 c0             	xor    %r8d,%r8d
    8208:	43 8b 3c 84          	mov    (%r12,%r8,4),%edi
    820c:	4c 89 da             	mov    %r11,%rdx
    820f:	31 c9                	xor    %ecx,%ecx
    8211:	44 89 c8             	mov    %r9d,%eax
    8214:	4c 8b 3a             	mov    (%rdx),%r15
    8217:	48 83 c2 08          	add    $0x8,%rdx
    821b:	d3 e8                	shr    %cl,%eax
    821d:	83 c1 08             	add    $0x8,%ecx
    8220:	0f b6 c0             	movzbl %al,%eax
    8223:	09 f8                	or     %edi,%eax
    8225:	48 98                	cltq   
    8227:	41 33 34 87          	xor    (%r15,%rax,4),%esi
    822b:	83 f9 20             	cmp    $0x20,%ecx
    822e:	75 e1                	jne    8211 <galois_w32_region_multiply+0xa1>
    8230:	49 83 c0 01          	add    $0x1,%r8
    8234:	49 83 c3 08          	add    $0x8,%r11
    8238:	49 83 f8 04          	cmp    $0x4,%r8
    823c:	75 ca                	jne    8208 <galois_w32_region_multiply+0x98>
    823e:	42 89 74 ad 00       	mov    %esi,0x0(%rbp,%r13,4)
    8243:	49 8d 45 01          	lea    0x1(%r13),%rax
    8247:	4d 39 ee             	cmp    %r13,%r14
    824a:	74 74                	je     82c0 <galois_w32_region_multiply+0x150>
    824c:	49 89 c5             	mov    %rax,%r13
    824f:	eb a7                	jmp    81f8 <galois_w32_region_multiply+0x88>
    8251:	45 8d 77 ff          	lea    -0x1(%r15),%r14d
    8255:	45 31 ed             	xor    %r13d,%r13d
    8258:	45 85 ff             	test   %r15d,%r15d
    825b:	7e 63                	jle    82c0 <galois_w32_region_multiply+0x150>
    825d:	0f 1f 00             	nopl   (%rax)
    8260:	46 8b 0c ab          	mov    (%rbx,%r13,4),%r9d
    8264:	4c 8d 1d 15 8f 00 00 	lea    0x8f15(%rip),%r11        # 11180 <galois_split_w8>
    826b:	45 31 c0             	xor    %r8d,%r8d
    826e:	31 f6                	xor    %esi,%esi
    8270:	43 8b 3c 84          	mov    (%r12,%r8,4),%edi
    8274:	4c 89 da             	mov    %r11,%rdx
    8277:	31 c9                	xor    %ecx,%ecx
    8279:	44 89 c8             	mov    %r9d,%eax
    827c:	4c 8b 3a             	mov    (%rdx),%r15
    827f:	48 83 c2 08          	add    $0x8,%rdx
    8283:	d3 e8                	shr    %cl,%eax
    8285:	83 c1 08             	add    $0x8,%ecx
    8288:	0f b6 c0             	movzbl %al,%eax
    828b:	09 f8                	or     %edi,%eax
    828d:	48 98                	cltq   
    828f:	41 33 34 87          	xor    (%r15,%rax,4),%esi
    8293:	83 f9 20             	cmp    $0x20,%ecx
    8296:	75 e1                	jne    8279 <galois_w32_region_multiply+0x109>
    8298:	49 83 c0 01          	add    $0x1,%r8
    829c:	49 83 c3 08          	add    $0x8,%r11
    82a0:	49 83 f8 04          	cmp    $0x4,%r8
    82a4:	75 ca                	jne    8270 <galois_w32_region_multiply+0x100>
    82a6:	42 31 74 ad 00       	xor    %esi,0x0(%rbp,%r13,4)
    82ab:	49 8d 45 01          	lea    0x1(%r13),%rax
    82af:	4d 39 ee             	cmp    %r13,%r14
    82b2:	74 0c                	je     82c0 <galois_w32_region_multiply+0x150>
    82b4:	49 89 c5             	mov    %rax,%r13
    82b7:	eb a7                	jmp    8260 <galois_w32_region_multiply+0xf0>
    82b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    82c0:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    82c5:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    82cc:	00 00 
    82ce:	75 48                	jne    8318 <galois_w32_region_multiply+0x1a8>
    82d0:	48 83 c4 28          	add    $0x28,%rsp
    82d4:	5b                   	pop    %rbx
    82d5:	5d                   	pop    %rbp
    82d6:	41 5c                	pop    %r12
    82d8:	41 5d                	pop    %r13
    82da:	41 5e                	pop    %r14
    82dc:	41 5f                	pop    %r15
    82de:	c3                   	retq   
    82df:	bf 08 00 00 00       	mov    $0x8,%edi
    82e4:	e8 37 fd ff ff       	callq  8020 <galois_create_split_w8_tables>
    82e9:	85 c0                	test   %eax,%eax
    82eb:	0f 89 c9 fe ff ff    	jns    81ba <galois_w32_region_multiply+0x4a>
    82f1:	48 8b 0d 48 8e 00 00 	mov    0x8e48(%rip),%rcx        # 11140 <stderr@@GLIBC_2.2.5>
    82f8:	ba 47 00 00 00       	mov    $0x47,%edx
    82fd:	be 01 00 00 00       	mov    $0x1,%esi
    8302:	48 8d 3d cf 26 00 00 	lea    0x26cf(%rip),%rdi        # a9d8 <__PRETTY_FUNCTION__.5230+0x307>
    8309:	e8 52 91 ff ff       	callq  1460 <fwrite@plt>
    830e:	bf 01 00 00 00       	mov    $0x1,%edi
    8313:	e8 38 91 ff ff       	callq  1450 <exit@plt>
    8318:	e8 e3 8f ff ff       	callq  1300 <__stack_chk_fail@plt>
    831d:	0f 1f 00             	nopl   (%rax)

0000000000008320 <galois_split_w8_multiply>:
    8320:	f3 0f 1e fa          	endbr64 
    8324:	41 56                	push   %r14
    8326:	45 31 c9             	xor    %r9d,%r9d
    8329:	45 31 db             	xor    %r11d,%r11d
    832c:	53                   	push   %rbx
    832d:	48 8d 1d 4c 8e 00 00 	lea    0x8e4c(%rip),%rbx        # 11180 <galois_split_w8>
    8334:	42 8d 0c dd 00 00 00 	lea    0x0(,%r11,8),%ecx
    833b:	00 
    833c:	41 89 f8             	mov    %edi,%r8d
    833f:	48 89 da             	mov    %rbx,%rdx
    8342:	41 d3 f8             	sar    %cl,%r8d
    8345:	31 c9                	xor    %ecx,%ecx
    8347:	41 c1 e0 08          	shl    $0x8,%r8d
    834b:	45 0f b7 c0          	movzwl %r8w,%r8d
    834f:	89 f0                	mov    %esi,%eax
    8351:	4c 8b 32             	mov    (%rdx),%r14
    8354:	48 83 c2 08          	add    $0x8,%rdx
    8358:	d3 f8                	sar    %cl,%eax
    835a:	83 c1 08             	add    $0x8,%ecx
    835d:	0f b6 c0             	movzbl %al,%eax
    8360:	44 09 c0             	or     %r8d,%eax
    8363:	48 98                	cltq   
    8365:	45 33 0c 86          	xor    (%r14,%rax,4),%r9d
    8369:	83 f9 20             	cmp    $0x20,%ecx
    836c:	75 e1                	jne    834f <galois_split_w8_multiply+0x2f>
    836e:	41 83 c3 01          	add    $0x1,%r11d
    8372:	48 83 c3 08          	add    $0x8,%rbx
    8376:	41 83 fb 04          	cmp    $0x4,%r11d
    837a:	75 b8                	jne    8334 <galois_split_w8_multiply+0x14>
    837c:	44 89 c8             	mov    %r9d,%eax
    837f:	5b                   	pop    %rbx
    8380:	41 5e                	pop    %r14
    8382:	c3                   	retq   
    8383:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    838a:	00 00 00 00 
    838e:	66 90                	xchg   %ax,%ax

0000000000008390 <galois_single_multiply.part.0>:
    8390:	41 55                	push   %r13
    8392:	48 8d 05 a7 28 00 00 	lea    0x28a7(%rip),%rax        # ac40 <mult_type>
    8399:	41 54                	push   %r12
    839b:	55                   	push   %rbp
    839c:	48 63 ef             	movslq %edi,%rbp
    839f:	53                   	push   %rbx
    83a0:	48 63 da             	movslq %edx,%rbx
    83a3:	49 89 dc             	mov    %rbx,%r12
    83a6:	48 83 ec 18          	sub    $0x18,%rsp
    83aa:	8b 04 98             	mov    (%rax,%rbx,4),%eax
    83ad:	83 f8 0b             	cmp    $0xb,%eax
    83b0:	74 66                	je     8418 <galois_single_multiply.part.0+0x88>
    83b2:	83 f8 0d             	cmp    $0xd,%eax
    83b5:	74 29                	je     83e0 <galois_single_multiply.part.0+0x50>
    83b7:	83 f8 0e             	cmp    $0xe,%eax
    83ba:	0f 84 b0 00 00 00    	je     8470 <galois_single_multiply.part.0+0xe0>
    83c0:	83 f8 0c             	cmp    $0xc,%eax
    83c3:	0f 85 2c 01 00 00    	jne    84f5 <galois_single_multiply.part.0+0x165>
    83c9:	48 83 c4 18          	add    $0x18,%rsp
    83cd:	89 da                	mov    %ebx,%edx
    83cf:	89 ef                	mov    %ebp,%edi
    83d1:	5b                   	pop    %rbx
    83d2:	5d                   	pop    %rbp
    83d3:	41 5c                	pop    %r12
    83d5:	41 5d                	pop    %r13
    83d7:	e9 94 f3 ff ff       	jmpq   7770 <galois_shift_multiply>
    83dc:	0f 1f 40 00          	nopl   0x0(%rax)
    83e0:	4c 8d 2d 39 91 00 00 	lea    0x9139(%rip),%r13        # 11520 <galois_log_tables>
    83e7:	49 8b 54 dd 00       	mov    0x0(%r13,%rbx,8),%rdx
    83ec:	48 85 d2             	test   %rdx,%rdx
    83ef:	74 57                	je     8448 <galois_single_multiply.part.0+0xb8>
    83f1:	48 63 f6             	movslq %esi,%rsi
    83f4:	8b 04 b2             	mov    (%rdx,%rsi,4),%eax
    83f7:	03 04 aa             	add    (%rdx,%rbp,4),%eax
    83fa:	48 8d 15 ff 8f 00 00 	lea    0x8fff(%rip),%rdx        # 11400 <galois_ilog_tables>
    8401:	48 8b 14 da          	mov    (%rdx,%rbx,8),%rdx
    8405:	48 98                	cltq   
    8407:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    840a:	48 83 c4 18          	add    $0x18,%rsp
    840e:	5b                   	pop    %rbx
    840f:	5d                   	pop    %rbp
    8410:	41 5c                	pop    %r12
    8412:	41 5d                	pop    %r13
    8414:	c3                   	retq   
    8415:	0f 1f 00             	nopl   (%rax)
    8418:	4c 8d 2d c1 8e 00 00 	lea    0x8ec1(%rip),%r13        # 112e0 <galois_mult_tables>
    841f:	49 8b 44 dd 00       	mov    0x0(%r13,%rbx,8),%rax
    8424:	48 85 c0             	test   %rax,%rax
    8427:	74 67                	je     8490 <galois_single_multiply.part.0+0x100>
    8429:	89 ef                	mov    %ebp,%edi
    842b:	44 89 e1             	mov    %r12d,%ecx
    842e:	d3 e7                	shl    %cl,%edi
    8430:	09 f7                	or     %esi,%edi
    8432:	48 63 ff             	movslq %edi,%rdi
    8435:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
    8438:	48 83 c4 18          	add    $0x18,%rsp
    843c:	5b                   	pop    %rbx
    843d:	5d                   	pop    %rbp
    843e:	41 5c                	pop    %r12
    8440:	41 5d                	pop    %r13
    8442:	c3                   	retq   
    8443:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8448:	89 74 24 0c          	mov    %esi,0xc(%rsp)
    844c:	83 fb 1e             	cmp    $0x1e,%ebx
    844f:	0f 8f 94 00 00 00    	jg     84e9 <galois_single_multiply.part.0+0x159>
    8455:	89 df                	mov    %ebx,%edi
    8457:	e8 b4 ed ff ff       	callq  7210 <galois_create_log_tables.part.0>
    845c:	85 c0                	test   %eax,%eax
    845e:	0f 88 85 00 00 00    	js     84e9 <galois_single_multiply.part.0+0x159>
    8464:	49 8b 54 dd 00       	mov    0x0(%r13,%rbx,8),%rdx
    8469:	8b 74 24 0c          	mov    0xc(%rsp),%esi
    846d:	eb 82                	jmp    83f1 <galois_single_multiply.part.0+0x61>
    846f:	90                   	nop
    8470:	48 83 3d 08 8d 00 00 	cmpq   $0x0,0x8d08(%rip)        # 11180 <galois_split_w8>
    8477:	00 
    8478:	74 36                	je     84b0 <galois_single_multiply.part.0+0x120>
    847a:	48 83 c4 18          	add    $0x18,%rsp
    847e:	89 ef                	mov    %ebp,%edi
    8480:	5b                   	pop    %rbx
    8481:	5d                   	pop    %rbp
    8482:	41 5c                	pop    %r12
    8484:	41 5d                	pop    %r13
    8486:	e9 95 fe ff ff       	jmpq   8320 <galois_split_w8_multiply>
    848b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8490:	89 df                	mov    %ebx,%edi
    8492:	89 74 24 0c          	mov    %esi,0xc(%rsp)
    8496:	e8 d5 ef ff ff       	callq  7470 <galois_create_mult_tables>
    849b:	85 c0                	test   %eax,%eax
    849d:	78 61                	js     8500 <galois_single_multiply.part.0+0x170>
    849f:	49 8b 44 dd 00       	mov    0x0(%r13,%rbx,8),%rax
    84a4:	8b 74 24 0c          	mov    0xc(%rsp),%esi
    84a8:	e9 7c ff ff ff       	jmpq   8429 <galois_single_multiply.part.0+0x99>
    84ad:	0f 1f 00             	nopl   (%rax)
    84b0:	31 c0                	xor    %eax,%eax
    84b2:	89 74 24 0c          	mov    %esi,0xc(%rsp)
    84b6:	e8 65 fb ff ff       	callq  8020 <galois_create_split_w8_tables>
    84bb:	8b 74 24 0c          	mov    0xc(%rsp),%esi
    84bf:	85 c0                	test   %eax,%eax
    84c1:	79 b7                	jns    847a <galois_single_multiply.part.0+0xea>
    84c3:	89 d9                	mov    %ebx,%ecx
    84c5:	48 8d 15 bc 25 00 00 	lea    0x25bc(%rip),%rdx        # aa88 <__PRETTY_FUNCTION__.5230+0x3b7>
    84cc:	48 8b 3d 6d 8c 00 00 	mov    0x8c6d(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    84d3:	be 01 00 00 00       	mov    $0x1,%esi
    84d8:	31 c0                	xor    %eax,%eax
    84da:	e8 91 8f ff ff       	callq  1470 <__fprintf_chk@plt>
    84df:	bf 01 00 00 00       	mov    $0x1,%edi
    84e4:	e8 67 8f ff ff       	callq  1450 <exit@plt>
    84e9:	44 89 e1             	mov    %r12d,%ecx
    84ec:	48 8d 15 65 25 00 00 	lea    0x2565(%rip),%rdx        # aa58 <__PRETTY_FUNCTION__.5230+0x387>
    84f3:	eb d7                	jmp    84cc <galois_single_multiply.part.0+0x13c>
    84f5:	89 d9                	mov    %ebx,%ecx
    84f7:	48 8d 15 c2 25 00 00 	lea    0x25c2(%rip),%rdx        # aac0 <__PRETTY_FUNCTION__.5230+0x3ef>
    84fe:	eb cc                	jmp    84cc <galois_single_multiply.part.0+0x13c>
    8500:	89 d9                	mov    %ebx,%ecx
    8502:	48 8d 15 17 25 00 00 	lea    0x2517(%rip),%rdx        # aa20 <__PRETTY_FUNCTION__.5230+0x34f>
    8509:	eb c1                	jmp    84cc <galois_single_multiply.part.0+0x13c>
    850b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000008510 <galois_single_multiply>:
    8510:	f3 0f 1e fa          	endbr64 
    8514:	85 ff                	test   %edi,%edi
    8516:	74 10                	je     8528 <galois_single_multiply+0x18>
    8518:	85 f6                	test   %esi,%esi
    851a:	74 0c                	je     8528 <galois_single_multiply+0x18>
    851c:	e9 6f fe ff ff       	jmpq   8390 <galois_single_multiply.part.0>
    8521:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8528:	31 c0                	xor    %eax,%eax
    852a:	c3                   	retq   
    852b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000008530 <galois_single_divide>:
    8530:	f3 0f 1e fa          	endbr64 
    8534:	41 56                	push   %r14
    8536:	48 8d 05 03 27 00 00 	lea    0x2703(%rip),%rax        # ac40 <mult_type>
    853d:	41 55                	push   %r13
    853f:	41 89 f5             	mov    %esi,%r13d
    8542:	41 54                	push   %r12
    8544:	55                   	push   %rbp
    8545:	48 63 ef             	movslq %edi,%rbp
    8548:	53                   	push   %rbx
    8549:	48 63 da             	movslq %edx,%rbx
    854c:	8b 04 98             	mov    (%rax,%rbx,4),%eax
    854f:	49 89 dc             	mov    %rbx,%r12
    8552:	83 f8 0b             	cmp    $0xb,%eax
    8555:	74 71                	je     85c8 <galois_single_divide+0x98>
    8557:	83 f8 0d             	cmp    $0xd,%eax
    855a:	74 24                	je     8580 <galois_single_divide+0x50>
    855c:	85 f6                	test   %esi,%esi
    855e:	0f 84 e5 00 00 00    	je     8649 <galois_single_divide+0x119>
    8564:	85 ed                	test   %ebp,%ebp
    8566:	0f 85 84 00 00 00    	jne    85f0 <galois_single_divide+0xc0>
    856c:	31 c0                	xor    %eax,%eax
    856e:	5b                   	pop    %rbx
    856f:	5d                   	pop    %rbp
    8570:	41 5c                	pop    %r12
    8572:	41 5d                	pop    %r13
    8574:	41 5e                	pop    %r14
    8576:	c3                   	retq   
    8577:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    857e:	00 00 
    8580:	85 f6                	test   %esi,%esi
    8582:	0f 84 c1 00 00 00    	je     8649 <galois_single_divide+0x119>
    8588:	85 ed                	test   %ebp,%ebp
    858a:	74 e0                	je     856c <galois_single_divide+0x3c>
    858c:	4c 8d 35 8d 8f 00 00 	lea    0x8f8d(%rip),%r14        # 11520 <galois_log_tables>
    8593:	49 8b 14 de          	mov    (%r14,%rbx,8),%rdx
    8597:	48 85 d2             	test   %rdx,%rdx
    859a:	0f 84 90 00 00 00    	je     8630 <galois_single_divide+0x100>
    85a0:	49 63 f5             	movslq %r13d,%rsi
    85a3:	8b 04 aa             	mov    (%rdx,%rbp,4),%eax
    85a6:	2b 04 b2             	sub    (%rdx,%rsi,4),%eax
    85a9:	48 8d 15 50 8e 00 00 	lea    0x8e50(%rip),%rdx        # 11400 <galois_ilog_tables>
    85b0:	48 8b 14 da          	mov    (%rdx,%rbx,8),%rdx
    85b4:	48 98                	cltq   
    85b6:	5b                   	pop    %rbx
    85b7:	5d                   	pop    %rbp
    85b8:	41 5c                	pop    %r12
    85ba:	8b 04 82             	mov    (%rdx,%rax,4),%eax
    85bd:	41 5d                	pop    %r13
    85bf:	41 5e                	pop    %r14
    85c1:	c3                   	retq   
    85c2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    85c8:	4c 8d 35 f1 8b 00 00 	lea    0x8bf1(%rip),%r14        # 111c0 <galois_div_tables>
    85cf:	49 8b 04 de          	mov    (%r14,%rbx,8),%rax
    85d3:	48 85 c0             	test   %rax,%rax
    85d6:	74 40                	je     8618 <galois_single_divide+0xe8>
    85d8:	44 89 e1             	mov    %r12d,%ecx
    85db:	5b                   	pop    %rbx
    85dc:	d3 e5                	shl    %cl,%ebp
    85de:	44 09 ed             	or     %r13d,%ebp
    85e1:	48 63 ed             	movslq %ebp,%rbp
    85e4:	8b 04 a8             	mov    (%rax,%rbp,4),%eax
    85e7:	5d                   	pop    %rbp
    85e8:	41 5c                	pop    %r12
    85ea:	41 5d                	pop    %r13
    85ec:	41 5e                	pop    %r14
    85ee:	c3                   	retq   
    85ef:	90                   	nop
    85f0:	89 de                	mov    %ebx,%esi
    85f2:	44 89 ef             	mov    %r13d,%edi
    85f5:	e8 96 00 00 00       	callq  8690 <galois_inverse>
    85fa:	89 c6                	mov    %eax,%esi
    85fc:	85 c0                	test   %eax,%eax
    85fe:	0f 84 68 ff ff ff    	je     856c <galois_single_divide+0x3c>
    8604:	89 da                	mov    %ebx,%edx
    8606:	89 ef                	mov    %ebp,%edi
    8608:	5b                   	pop    %rbx
    8609:	5d                   	pop    %rbp
    860a:	41 5c                	pop    %r12
    860c:	41 5d                	pop    %r13
    860e:	41 5e                	pop    %r14
    8610:	e9 7b fd ff ff       	jmpq   8390 <galois_single_multiply.part.0>
    8615:	0f 1f 00             	nopl   (%rax)
    8618:	89 df                	mov    %ebx,%edi
    861a:	e8 51 ee ff ff       	callq  7470 <galois_create_mult_tables>
    861f:	85 c0                	test   %eax,%eax
    8621:	78 57                	js     867a <galois_single_divide+0x14a>
    8623:	49 8b 04 de          	mov    (%r14,%rbx,8),%rax
    8627:	eb af                	jmp    85d8 <galois_single_divide+0xa8>
    8629:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8630:	83 fb 1e             	cmp    $0x1e,%ebx
    8633:	7f 1e                	jg     8653 <galois_single_divide+0x123>
    8635:	89 df                	mov    %ebx,%edi
    8637:	e8 d4 eb ff ff       	callq  7210 <galois_create_log_tables.part.0>
    863c:	85 c0                	test   %eax,%eax
    863e:	78 13                	js     8653 <galois_single_divide+0x123>
    8640:	49 8b 14 de          	mov    (%r14,%rbx,8),%rdx
    8644:	e9 57 ff ff ff       	jmpq   85a0 <galois_single_divide+0x70>
    8649:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    864e:	e9 1b ff ff ff       	jmpq   856e <galois_single_divide+0x3e>
    8653:	44 89 e1             	mov    %r12d,%ecx
    8656:	48 8d 15 fb 23 00 00 	lea    0x23fb(%rip),%rdx        # aa58 <__PRETTY_FUNCTION__.5230+0x387>
    865d:	48 8b 3d dc 8a 00 00 	mov    0x8adc(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    8664:	be 01 00 00 00       	mov    $0x1,%esi
    8669:	31 c0                	xor    %eax,%eax
    866b:	e8 00 8e ff ff       	callq  1470 <__fprintf_chk@plt>
    8670:	bf 01 00 00 00       	mov    $0x1,%edi
    8675:	e8 d6 8d ff ff       	callq  1450 <exit@plt>
    867a:	89 d9                	mov    %ebx,%ecx
    867c:	48 8d 15 9d 23 00 00 	lea    0x239d(%rip),%rdx        # aa20 <__PRETTY_FUNCTION__.5230+0x34f>
    8683:	eb d8                	jmp    865d <galois_single_divide+0x12d>
    8685:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    868c:	00 00 00 00 

0000000000008690 <galois_inverse>:
    8690:	f3 0f 1e fa          	endbr64 
    8694:	89 f2                	mov    %esi,%edx
    8696:	85 ff                	test   %edi,%edi
    8698:	74 2e                	je     86c8 <galois_inverse+0x38>
    869a:	48 63 c6             	movslq %esi,%rax
    869d:	48 8d 0d 9c 25 00 00 	lea    0x259c(%rip),%rcx        # ac40 <mult_type>
    86a4:	8b 04 81             	mov    (%rcx,%rax,4),%eax
    86a7:	83 e0 fd             	and    $0xfffffffd,%eax
    86aa:	83 f8 0c             	cmp    $0xc,%eax
    86ad:	74 11                	je     86c0 <galois_inverse+0x30>
    86af:	89 fe                	mov    %edi,%esi
    86b1:	bf 01 00 00 00       	mov    $0x1,%edi
    86b6:	e9 75 fe ff ff       	jmpq   8530 <galois_single_divide>
    86bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    86c0:	e9 bb f6 ff ff       	jmpq   7d80 <galois_shift_inverse>
    86c5:	0f 1f 00             	nopl   (%rax)
    86c8:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    86cd:	c3                   	retq   
    86ce:	66 90                	xchg   %ax,%ax

00000000000086d0 <reed_sol_r6_coding_matrix>:
    86d0:	f3 0f 1e fa          	endbr64 
    86d4:	41 55                	push   %r13
    86d6:	8d 46 f8             	lea    -0x8(%rsi),%eax
    86d9:	4c 63 ef             	movslq %edi,%r13
    86dc:	41 54                	push   %r12
    86de:	55                   	push   %rbp
    86df:	89 f5                	mov    %esi,%ebp
    86e1:	53                   	push   %rbx
    86e2:	48 83 ec 08          	sub    $0x8,%rsp
    86e6:	83 e0 f7             	and    $0xfffffff7,%eax
    86e9:	74 09                	je     86f4 <reed_sol_r6_coding_matrix+0x24>
    86eb:	83 fe 20             	cmp    $0x20,%esi
    86ee:	0f 85 8c 00 00 00    	jne    8780 <reed_sol_r6_coding_matrix+0xb0>
    86f4:	43 8d 7c 2d 00       	lea    0x0(%r13,%r13,1),%edi
    86f9:	48 63 ff             	movslq %edi,%rdi
    86fc:	48 c1 e7 02          	shl    $0x2,%rdi
    8700:	e8 eb 8c ff ff       	callq  13f0 <malloc@plt>
    8705:	49 89 c4             	mov    %rax,%r12
    8708:	48 85 c0             	test   %rax,%rax
    870b:	74 73                	je     8780 <reed_sol_r6_coding_matrix+0xb0>
    870d:	45 85 ed             	test   %r13d,%r13d
    8710:	0f 8e 82 00 00 00    	jle    8798 <reed_sol_r6_coding_matrix+0xc8>
    8716:	41 8d 55 ff          	lea    -0x1(%r13),%edx
    871a:	48 8d 54 90 04       	lea    0x4(%rax,%rdx,4),%rdx
    871f:	90                   	nop
    8720:	c7 00 01 00 00 00    	movl   $0x1,(%rax)
    8726:	48 83 c0 04          	add    $0x4,%rax
    872a:	48 39 d0             	cmp    %rdx,%rax
    872d:	75 f1                	jne    8720 <reed_sol_r6_coding_matrix+0x50>
    872f:	49 63 c5             	movslq %r13d,%rax
    8732:	49 8d 1c 84          	lea    (%r12,%rax,4),%rbx
    8736:	c7 03 01 00 00 00    	movl   $0x1,(%rbx)
    873c:	41 83 fd 01          	cmp    $0x1,%r13d
    8740:	7e 2f                	jle    8771 <reed_sol_r6_coding_matrix+0xa1>
    8742:	41 8d 55 fe          	lea    -0x2(%r13),%edx
    8746:	bf 01 00 00 00       	mov    $0x1,%edi
    874b:	48 8d 44 10 01       	lea    0x1(%rax,%rdx,1),%rax
    8750:	4d 8d 2c 84          	lea    (%r12,%rax,4),%r13
    8754:	0f 1f 40 00          	nopl   0x0(%rax)
    8758:	89 ea                	mov    %ebp,%edx
    875a:	be 02 00 00 00       	mov    $0x2,%esi
    875f:	48 83 c3 04          	add    $0x4,%rbx
    8763:	e8 a8 fd ff ff       	callq  8510 <galois_single_multiply>
    8768:	89 03                	mov    %eax,(%rbx)
    876a:	89 c7                	mov    %eax,%edi
    876c:	4c 39 eb             	cmp    %r13,%rbx
    876f:	75 e7                	jne    8758 <reed_sol_r6_coding_matrix+0x88>
    8771:	48 83 c4 08          	add    $0x8,%rsp
    8775:	4c 89 e0             	mov    %r12,%rax
    8778:	5b                   	pop    %rbx
    8779:	5d                   	pop    %rbp
    877a:	41 5c                	pop    %r12
    877c:	41 5d                	pop    %r13
    877e:	c3                   	retq   
    877f:	90                   	nop
    8780:	48 83 c4 08          	add    $0x8,%rsp
    8784:	45 31 e4             	xor    %r12d,%r12d
    8787:	5b                   	pop    %rbx
    8788:	4c 89 e0             	mov    %r12,%rax
    878b:	5d                   	pop    %rbp
    878c:	41 5c                	pop    %r12
    878e:	41 5d                	pop    %r13
    8790:	c3                   	retq   
    8791:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8798:	42 c7 04 a8 01 00 00 	movl   $0x1,(%rax,%r13,4)
    879f:	00 
    87a0:	eb cf                	jmp    8771 <reed_sol_r6_coding_matrix+0xa1>
    87a2:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    87a9:	00 00 00 00 
    87ad:	0f 1f 00             	nopl   (%rax)

00000000000087b0 <reed_sol_galois_w32_region_multby_2>:
    87b0:	f3 0f 1e fa          	endbr64 
    87b4:	55                   	push   %rbp
    87b5:	89 f5                	mov    %esi,%ebp
    87b7:	53                   	push   %rbx
    87b8:	48 89 fb             	mov    %rdi,%rbx
    87bb:	48 83 ec 08          	sub    $0x8,%rsp
    87bf:	83 3d c2 58 00 00 ff 	cmpl   $0xffffffff,0x58c2(%rip)        # e088 <prim32>
    87c6:	74 40                	je     8808 <reed_sol_galois_w32_region_multby_2+0x58>
    87c8:	48 63 f5             	movslq %ebp,%rsi
    87cb:	8b 3d b7 58 00 00    	mov    0x58b7(%rip),%edi        # e088 <prim32>
    87d1:	48 01 de             	add    %rbx,%rsi
    87d4:	48 39 f3             	cmp    %rsi,%rbx
    87d7:	73 21                	jae    87fa <reed_sol_galois_w32_region_multby_2+0x4a>
    87d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    87e0:	8b 13                	mov    (%rbx),%edx
    87e2:	8d 04 12             	lea    (%rdx,%rdx,1),%eax
    87e5:	89 c1                	mov    %eax,%ecx
    87e7:	31 f9                	xor    %edi,%ecx
    87e9:	85 d2                	test   %edx,%edx
    87eb:	0f 48 c1             	cmovs  %ecx,%eax
    87ee:	48 83 c3 04          	add    $0x4,%rbx
    87f2:	89 43 fc             	mov    %eax,-0x4(%rbx)
    87f5:	48 39 de             	cmp    %rbx,%rsi
    87f8:	77 e6                	ja     87e0 <reed_sol_galois_w32_region_multby_2+0x30>
    87fa:	48 83 c4 08          	add    $0x8,%rsp
    87fe:	5b                   	pop    %rbx
    87ff:	5d                   	pop    %rbp
    8800:	c3                   	retq   
    8801:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8808:	ba 20 00 00 00       	mov    $0x20,%edx
    880d:	be 02 00 00 00       	mov    $0x2,%esi
    8812:	bf 00 00 00 80       	mov    $0x80000000,%edi
    8817:	e8 f4 fc ff ff       	callq  8510 <galois_single_multiply>
    881c:	89 05 66 58 00 00    	mov    %eax,0x5866(%rip)        # e088 <prim32>
    8822:	eb a4                	jmp    87c8 <reed_sol_galois_w32_region_multby_2+0x18>
    8824:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    882b:	00 00 00 00 
    882f:	90                   	nop

0000000000008830 <reed_sol_galois_w08_region_multby_2>:
    8830:	f3 0f 1e fa          	endbr64 
    8834:	55                   	push   %rbp
    8835:	89 f5                	mov    %esi,%ebp
    8837:	53                   	push   %rbx
    8838:	48 89 fb             	mov    %rdi,%rbx
    883b:	48 83 ec 08          	sub    $0x8,%rsp
    883f:	83 3d 3e 58 00 00 ff 	cmpl   $0xffffffff,0x583e(%rip)        # e084 <prim08>
    8846:	74 58                	je     88a0 <reed_sol_galois_w08_region_multby_2+0x70>
    8848:	48 63 f5             	movslq %ebp,%rsi
    884b:	48 01 de             	add    %rbx,%rsi
    884e:	48 39 f3             	cmp    %rsi,%rbx
    8851:	73 41                	jae    8894 <reed_sol_galois_w08_region_multby_2+0x64>
    8853:	44 8b 0d 26 58 00 00 	mov    0x5826(%rip),%r9d        # e080 <mask08_1>
    885a:	44 8b 05 1b 58 00 00 	mov    0x581b(%rip),%r8d        # e07c <mask08_2>
    8861:	8b 3d 1d 58 00 00    	mov    0x581d(%rip),%edi        # e084 <prim08>
    8867:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    886e:	00 00 
    8870:	8b 13                	mov    (%rbx),%edx
    8872:	48 83 c3 04          	add    $0x4,%rbx
    8876:	89 d1                	mov    %edx,%ecx
    8878:	01 d2                	add    %edx,%edx
    887a:	44 21 c1             	and    %r8d,%ecx
    887d:	44 21 ca             	and    %r9d,%edx
    8880:	8d 04 09             	lea    (%rcx,%rcx,1),%eax
    8883:	c1 e9 07             	shr    $0x7,%ecx
    8886:	29 c8                	sub    %ecx,%eax
    8888:	21 f8                	and    %edi,%eax
    888a:	31 d0                	xor    %edx,%eax
    888c:	89 43 fc             	mov    %eax,-0x4(%rbx)
    888f:	48 39 de             	cmp    %rbx,%rsi
    8892:	77 dc                	ja     8870 <reed_sol_galois_w08_region_multby_2+0x40>
    8894:	48 83 c4 08          	add    $0x8,%rsp
    8898:	5b                   	pop    %rbx
    8899:	5d                   	pop    %rbp
    889a:	c3                   	retq   
    889b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    88a0:	ba 08 00 00 00       	mov    $0x8,%edx
    88a5:	be 02 00 00 00       	mov    $0x2,%esi
    88aa:	bf 80 00 00 00       	mov    $0x80,%edi
    88af:	e8 5c fc ff ff       	callq  8510 <galois_single_multiply>
    88b4:	c7 05 c6 57 00 00 00 	movl   $0x0,0x57c6(%rip)        # e084 <prim08>
    88bb:	00 00 00 
    88be:	85 c0                	test   %eax,%eax
    88c0:	74 13                	je     88d5 <reed_sol_galois_w08_region_multby_2+0xa5>
    88c2:	31 d2                	xor    %edx,%edx
    88c4:	0f 1f 40 00          	nopl   0x0(%rax)
    88c8:	09 c2                	or     %eax,%edx
    88ca:	c1 e0 08             	shl    $0x8,%eax
    88cd:	75 f9                	jne    88c8 <reed_sol_galois_w08_region_multby_2+0x98>
    88cf:	89 15 af 57 00 00    	mov    %edx,0x57af(%rip)        # e084 <prim08>
    88d5:	c7 05 a1 57 00 00 fe 	movl   $0xfefefefe,0x57a1(%rip)        # e080 <mask08_1>
    88dc:	fe fe fe 
    88df:	c7 05 93 57 00 00 80 	movl   $0x80808080,0x5793(%rip)        # e07c <mask08_2>
    88e6:	80 80 80 
    88e9:	e9 5a ff ff ff       	jmpq   8848 <reed_sol_galois_w08_region_multby_2+0x18>
    88ee:	66 90                	xchg   %ax,%ax

00000000000088f0 <reed_sol_galois_w16_region_multby_2>:
    88f0:	f3 0f 1e fa          	endbr64 
    88f4:	55                   	push   %rbp
    88f5:	89 f5                	mov    %esi,%ebp
    88f7:	53                   	push   %rbx
    88f8:	48 89 fb             	mov    %rdi,%rbx
    88fb:	48 83 ec 08          	sub    $0x8,%rsp
    88ff:	83 3d 72 57 00 00 ff 	cmpl   $0xffffffff,0x5772(%rip)        # e078 <prim16>
    8906:	74 58                	je     8960 <reed_sol_galois_w16_region_multby_2+0x70>
    8908:	48 63 f5             	movslq %ebp,%rsi
    890b:	48 01 de             	add    %rbx,%rsi
    890e:	48 39 f3             	cmp    %rsi,%rbx
    8911:	73 41                	jae    8954 <reed_sol_galois_w16_region_multby_2+0x64>
    8913:	44 8b 0d 5a 57 00 00 	mov    0x575a(%rip),%r9d        # e074 <mask16_1>
    891a:	44 8b 05 4f 57 00 00 	mov    0x574f(%rip),%r8d        # e070 <mask16_2>
    8921:	8b 3d 51 57 00 00    	mov    0x5751(%rip),%edi        # e078 <prim16>
    8927:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    892e:	00 00 
    8930:	8b 13                	mov    (%rbx),%edx
    8932:	48 83 c3 04          	add    $0x4,%rbx
    8936:	89 d1                	mov    %edx,%ecx
    8938:	01 d2                	add    %edx,%edx
    893a:	44 21 c1             	and    %r8d,%ecx
    893d:	44 21 ca             	and    %r9d,%edx
    8940:	8d 04 09             	lea    (%rcx,%rcx,1),%eax
    8943:	c1 e9 0f             	shr    $0xf,%ecx
    8946:	29 c8                	sub    %ecx,%eax
    8948:	21 f8                	and    %edi,%eax
    894a:	31 d0                	xor    %edx,%eax
    894c:	89 43 fc             	mov    %eax,-0x4(%rbx)
    894f:	48 39 de             	cmp    %rbx,%rsi
    8952:	77 dc                	ja     8930 <reed_sol_galois_w16_region_multby_2+0x40>
    8954:	48 83 c4 08          	add    $0x8,%rsp
    8958:	5b                   	pop    %rbx
    8959:	5d                   	pop    %rbp
    895a:	c3                   	retq   
    895b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8960:	ba 10 00 00 00       	mov    $0x10,%edx
    8965:	be 02 00 00 00       	mov    $0x2,%esi
    896a:	bf 00 80 00 00       	mov    $0x8000,%edi
    896f:	e8 9c fb ff ff       	callq  8510 <galois_single_multiply>
    8974:	c7 05 fa 56 00 00 00 	movl   $0x0,0x56fa(%rip)        # e078 <prim16>
    897b:	00 00 00 
    897e:	85 c0                	test   %eax,%eax
    8980:	74 13                	je     8995 <reed_sol_galois_w16_region_multby_2+0xa5>
    8982:	31 d2                	xor    %edx,%edx
    8984:	0f 1f 40 00          	nopl   0x0(%rax)
    8988:	09 c2                	or     %eax,%edx
    898a:	c1 e0 10             	shl    $0x10,%eax
    898d:	75 f9                	jne    8988 <reed_sol_galois_w16_region_multby_2+0x98>
    898f:	89 15 e3 56 00 00    	mov    %edx,0x56e3(%rip)        # e078 <prim16>
    8995:	c7 05 d5 56 00 00 fe 	movl   $0xfffefffe,0x56d5(%rip)        # e074 <mask16_1>
    899c:	ff fe ff 
    899f:	c7 05 c7 56 00 00 00 	movl   $0x80008000,0x56c7(%rip)        # e070 <mask16_2>
    89a6:	80 00 80 
    89a9:	e9 5a ff ff ff       	jmpq   8908 <reed_sol_galois_w16_region_multby_2+0x18>
    89ae:	66 90                	xchg   %ax,%ax

00000000000089b0 <reed_sol_r6_encode>:
    89b0:	f3 0f 1e fa          	endbr64 
    89b4:	41 57                	push   %r15
    89b6:	49 63 c0             	movslq %r8d,%rax
    89b9:	49 89 cf             	mov    %rcx,%r15
    89bc:	41 56                	push   %r14
    89be:	41 89 fe             	mov    %edi,%r14d
    89c1:	41 55                	push   %r13
    89c3:	49 89 d5             	mov    %rdx,%r13
    89c6:	41 54                	push   %r12
    89c8:	41 89 f4             	mov    %esi,%r12d
    89cb:	55                   	push   %rbp
    89cc:	53                   	push   %rbx
    89cd:	48 89 c3             	mov    %rax,%rbx
    89d0:	48 83 ec 28          	sub    $0x28,%rsp
    89d4:	48 8b 32             	mov    (%rdx),%rsi
    89d7:	48 8b 39             	mov    (%rcx),%rdi
    89da:	48 89 c2             	mov    %rax,%rdx
    89dd:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    89e2:	e8 c9 89 ff ff       	callq  13b0 <memcpy@plt>
    89e7:	49 63 c6             	movslq %r14d,%rax
    89ea:	49 8d 44 c5 f8       	lea    -0x8(%r13,%rax,8),%rax
    89ef:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    89f4:	41 8d 46 fe          	lea    -0x2(%r14),%eax
    89f8:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    89fc:	41 83 fe 01          	cmp    $0x1,%r14d
    8a00:	0f 8e ca 00 00 00    	jle    8ad0 <reed_sol_r6_encode+0x120>
    8a06:	89 c2                	mov    %eax,%edx
    8a08:	49 8d 6d 08          	lea    0x8(%r13),%rbp
    8a0c:	4d 8d 74 d5 10       	lea    0x10(%r13,%rdx,8),%r14
    8a11:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8a18:	49 8b 3f             	mov    (%r15),%rdi
    8a1b:	48 8b 75 00          	mov    0x0(%rbp),%rsi
    8a1f:	89 d9                	mov    %ebx,%ecx
    8a21:	48 83 c5 08          	add    $0x8,%rbp
    8a25:	48 89 fa             	mov    %rdi,%rdx
    8a28:	e8 b3 f5 ff ff       	callq  7fe0 <galois_region_xor>
    8a2d:	49 39 ee             	cmp    %rbp,%r14
    8a30:	75 e6                	jne    8a18 <reed_sol_r6_encode+0x68>
    8a32:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    8a37:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8a3b:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    8a40:	48 8b 30             	mov    (%rax),%rsi
    8a43:	e8 68 89 ff ff       	callq  13b0 <memcpy@plt>
    8a48:	48 63 6c 24 0c       	movslq 0xc(%rsp),%rbp
    8a4d:	41 83 fc 10          	cmp    $0x10,%r12d
    8a51:	74 1d                	je     8a70 <reed_sol_r6_encode+0xc0>
    8a53:	41 83 fc 20          	cmp    $0x20,%r12d
    8a57:	74 67                	je     8ac0 <reed_sol_r6_encode+0x110>
    8a59:	41 83 fc 08          	cmp    $0x8,%r12d
    8a5d:	74 51                	je     8ab0 <reed_sol_r6_encode+0x100>
    8a5f:	48 83 c4 28          	add    $0x28,%rsp
    8a63:	31 c0                	xor    %eax,%eax
    8a65:	5b                   	pop    %rbx
    8a66:	5d                   	pop    %rbp
    8a67:	41 5c                	pop    %r12
    8a69:	41 5d                	pop    %r13
    8a6b:	41 5e                	pop    %r14
    8a6d:	41 5f                	pop    %r15
    8a6f:	c3                   	retq   
    8a70:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8a74:	89 de                	mov    %ebx,%esi
    8a76:	e8 75 fe ff ff       	callq  88f0 <reed_sol_galois_w16_region_multby_2>
    8a7b:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8a7f:	49 8b 74 ed 00       	mov    0x0(%r13,%rbp,8),%rsi
    8a84:	89 d9                	mov    %ebx,%ecx
    8a86:	48 83 ed 01          	sub    $0x1,%rbp
    8a8a:	48 89 fa             	mov    %rdi,%rdx
    8a8d:	e8 4e f5 ff ff       	callq  7fe0 <galois_region_xor>
    8a92:	85 ed                	test   %ebp,%ebp
    8a94:	79 b7                	jns    8a4d <reed_sol_r6_encode+0x9d>
    8a96:	48 83 c4 28          	add    $0x28,%rsp
    8a9a:	b8 01 00 00 00       	mov    $0x1,%eax
    8a9f:	5b                   	pop    %rbx
    8aa0:	5d                   	pop    %rbp
    8aa1:	41 5c                	pop    %r12
    8aa3:	41 5d                	pop    %r13
    8aa5:	41 5e                	pop    %r14
    8aa7:	41 5f                	pop    %r15
    8aa9:	c3                   	retq   
    8aaa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    8ab0:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8ab4:	89 de                	mov    %ebx,%esi
    8ab6:	e8 75 fd ff ff       	callq  8830 <reed_sol_galois_w08_region_multby_2>
    8abb:	eb be                	jmp    8a7b <reed_sol_r6_encode+0xcb>
    8abd:	0f 1f 00             	nopl   (%rax)
    8ac0:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8ac4:	89 de                	mov    %ebx,%esi
    8ac6:	e8 e5 fc ff ff       	callq  87b0 <reed_sol_galois_w32_region_multby_2>
    8acb:	eb ae                	jmp    8a7b <reed_sol_r6_encode+0xcb>
    8acd:	0f 1f 00             	nopl   (%rax)
    8ad0:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    8ad5:	49 8b 7f 08          	mov    0x8(%r15),%rdi
    8ad9:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    8ade:	48 8b 30             	mov    (%rax),%rsi
    8ae1:	e8 ca 88 ff ff       	callq  13b0 <memcpy@plt>
    8ae6:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    8aea:	85 c0                	test   %eax,%eax
    8aec:	0f 89 56 ff ff ff    	jns    8a48 <reed_sol_r6_encode+0x98>
    8af2:	eb a2                	jmp    8a96 <reed_sol_r6_encode+0xe6>
    8af4:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    8afb:	00 00 00 00 
    8aff:	90                   	nop

0000000000008b00 <reed_sol_extended_vandermonde_matrix>:
    8b00:	f3 0f 1e fa          	endbr64 
    8b04:	41 57                	push   %r15
    8b06:	41 89 d7             	mov    %edx,%r15d
    8b09:	41 56                	push   %r14
    8b0b:	41 55                	push   %r13
    8b0d:	41 54                	push   %r12
    8b0f:	41 89 f4             	mov    %esi,%r12d
    8b12:	55                   	push   %rbp
    8b13:	53                   	push   %rbx
    8b14:	89 fb                	mov    %edi,%ebx
    8b16:	48 83 ec 18          	sub    $0x18,%rsp
    8b1a:	83 fa 1d             	cmp    $0x1d,%edx
    8b1d:	7f 19                	jg     8b38 <reed_sol_extended_vandermonde_matrix+0x38>
    8b1f:	b8 01 00 00 00       	mov    $0x1,%eax
    8b24:	89 d1                	mov    %edx,%ecx
    8b26:	d3 e0                	shl    %cl,%eax
    8b28:	39 f8                	cmp    %edi,%eax
    8b2a:	0f 8c 18 01 00 00    	jl     8c48 <reed_sol_extended_vandermonde_matrix+0x148>
    8b30:	39 f0                	cmp    %esi,%eax
    8b32:	0f 8c 10 01 00 00    	jl     8c48 <reed_sol_extended_vandermonde_matrix+0x148>
    8b38:	89 dd                	mov    %ebx,%ebp
    8b3a:	41 0f af ec          	imul   %r12d,%ebp
    8b3e:	48 63 fd             	movslq %ebp,%rdi
    8b41:	48 c1 e7 02          	shl    $0x2,%rdi
    8b45:	e8 a6 88 ff ff       	callq  13f0 <malloc@plt>
    8b4a:	49 89 c5             	mov    %rax,%r13
    8b4d:	48 85 c0             	test   %rax,%rax
    8b50:	0f 84 f2 00 00 00    	je     8c48 <reed_sol_extended_vandermonde_matrix+0x148>
    8b56:	c7 00 01 00 00 00    	movl   $0x1,(%rax)
    8b5c:	41 83 fc 01          	cmp    $0x1,%r12d
    8b60:	0f 8e e7 00 00 00    	jle    8c4d <reed_sol_extended_vandermonde_matrix+0x14d>
    8b66:	41 8d 54 24 fe       	lea    -0x2(%r12),%edx
    8b6b:	48 8d 40 04          	lea    0x4(%rax),%rax
    8b6f:	49 8d 54 95 08       	lea    0x8(%r13,%rdx,4),%rdx
    8b74:	0f 1f 40 00          	nopl   0x0(%rax)
    8b78:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
    8b7e:	48 83 c0 04          	add    $0x4,%rax
    8b82:	48 39 d0             	cmp    %rdx,%rax
    8b85:	75 f1                	jne    8b78 <reed_sol_extended_vandermonde_matrix+0x78>
    8b87:	83 fb 01             	cmp    $0x1,%ebx
    8b8a:	0f 84 a6 00 00 00    	je     8c36 <reed_sol_extended_vandermonde_matrix+0x136>
    8b90:	8d 43 ff             	lea    -0x1(%rbx),%eax
    8b93:	41 8d 74 24 ff       	lea    -0x1(%r12),%esi
    8b98:	89 44 24 08          	mov    %eax,0x8(%rsp)
    8b9c:	89 e8                	mov    %ebp,%eax
    8b9e:	44 29 e0             	sub    %r12d,%eax
    8ba1:	48 98                	cltq   
    8ba3:	49 8d 54 85 00       	lea    0x0(%r13,%rax,4),%rdx
    8ba8:	31 c0                	xor    %eax,%eax
    8baa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    8bb0:	c7 04 82 00 00 00 00 	movl   $0x0,(%rdx,%rax,4)
    8bb7:	48 83 c0 01          	add    $0x1,%rax
    8bbb:	39 c6                	cmp    %eax,%esi
    8bbd:	7f f1                	jg     8bb0 <reed_sol_extended_vandermonde_matrix+0xb0>
    8bbf:	83 ed 01             	sub    $0x1,%ebp
    8bc2:	48 63 ed             	movslq %ebp,%rbp
    8bc5:	41 c7 44 ad 00 01 00 	movl   $0x1,0x0(%r13,%rbp,4)
    8bcc:	00 00 
    8bce:	83 fb 02             	cmp    $0x2,%ebx
    8bd1:	74 63                	je     8c36 <reed_sol_extended_vandermonde_matrix+0x136>
    8bd3:	83 7c 24 08 01       	cmpl   $0x1,0x8(%rsp)
    8bd8:	7e 5c                	jle    8c36 <reed_sol_extended_vandermonde_matrix+0x136>
    8bda:	41 8d 44 24 ff       	lea    -0x1(%r12),%eax
    8bdf:	44 89 64 24 04       	mov    %r12d,0x4(%rsp)
    8be4:	bb 01 00 00 00       	mov    $0x1,%ebx
    8be9:	89 44 24 0c          	mov    %eax,0xc(%rsp)
    8bed:	0f 1f 00             	nopl   (%rax)
    8bf0:	45 85 e4             	test   %r12d,%r12d
    8bf3:	7e 33                	jle    8c28 <reed_sol_extended_vandermonde_matrix+0x128>
    8bf5:	48 63 54 24 04       	movslq 0x4(%rsp),%rdx
    8bfa:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    8bfe:	bf 01 00 00 00       	mov    $0x1,%edi
    8c03:	48 01 d0             	add    %rdx,%rax
    8c06:	4d 8d 74 95 00       	lea    0x0(%r13,%rdx,4),%r14
    8c0b:	49 8d 6c 85 04       	lea    0x4(%r13,%rax,4),%rbp
    8c10:	41 89 3e             	mov    %edi,(%r14)
    8c13:	44 89 fa             	mov    %r15d,%edx
    8c16:	89 de                	mov    %ebx,%esi
    8c18:	49 83 c6 04          	add    $0x4,%r14
    8c1c:	e8 ef f8 ff ff       	callq  8510 <galois_single_multiply>
    8c21:	89 c7                	mov    %eax,%edi
    8c23:	4c 39 f5             	cmp    %r14,%rbp
    8c26:	75 e8                	jne    8c10 <reed_sol_extended_vandermonde_matrix+0x110>
    8c28:	44 01 64 24 04       	add    %r12d,0x4(%rsp)
    8c2d:	83 c3 01             	add    $0x1,%ebx
    8c30:	39 5c 24 08          	cmp    %ebx,0x8(%rsp)
    8c34:	75 ba                	jne    8bf0 <reed_sol_extended_vandermonde_matrix+0xf0>
    8c36:	48 83 c4 18          	add    $0x18,%rsp
    8c3a:	4c 89 e8             	mov    %r13,%rax
    8c3d:	5b                   	pop    %rbx
    8c3e:	5d                   	pop    %rbp
    8c3f:	41 5c                	pop    %r12
    8c41:	41 5d                	pop    %r13
    8c43:	41 5e                	pop    %r14
    8c45:	41 5f                	pop    %r15
    8c47:	c3                   	retq   
    8c48:	45 31 ed             	xor    %r13d,%r13d
    8c4b:	eb e9                	jmp    8c36 <reed_sol_extended_vandermonde_matrix+0x136>
    8c4d:	83 fb 01             	cmp    $0x1,%ebx
    8c50:	74 e4                	je     8c36 <reed_sol_extended_vandermonde_matrix+0x136>
    8c52:	8d 43 ff             	lea    -0x1(%rbx),%eax
    8c55:	44 29 e5             	sub    %r12d,%ebp
    8c58:	89 44 24 08          	mov    %eax,0x8(%rsp)
    8c5c:	e9 61 ff ff ff       	jmpq   8bc2 <reed_sol_extended_vandermonde_matrix+0xc2>
    8c61:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    8c68:	00 00 00 00 
    8c6c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000008c70 <reed_sol_big_vandermonde_distribution_matrix>:
    8c70:	f3 0f 1e fa          	endbr64 
    8c74:	41 57                	push   %r15
    8c76:	41 56                	push   %r14
    8c78:	41 55                	push   %r13
    8c7a:	41 54                	push   %r12
    8c7c:	55                   	push   %rbp
    8c7d:	53                   	push   %rbx
    8c7e:	48 83 ec 78          	sub    $0x78,%rsp
    8c82:	89 7c 24 04          	mov    %edi,0x4(%rsp)
    8c86:	89 74 24 6c          	mov    %esi,0x6c(%rsp)
    8c8a:	39 fe                	cmp    %edi,%esi
    8c8c:	0f 8d fe 03 00 00    	jge    9090 <reed_sol_big_vandermonde_distribution_matrix+0x420>
    8c92:	89 f3                	mov    %esi,%ebx
    8c94:	41 89 d4             	mov    %edx,%r12d
    8c97:	e8 64 fe ff ff       	callq  8b00 <reed_sol_extended_vandermonde_matrix>
    8c9c:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    8ca1:	48 89 c1             	mov    %rax,%rcx
    8ca4:	48 85 c0             	test   %rax,%rax
    8ca7:	0f 84 e3 03 00 00    	je     9090 <reed_sol_big_vandermonde_distribution_matrix+0x420>
    8cad:	83 fb 01             	cmp    $0x1,%ebx
    8cb0:	0f 8e dc 01 00 00    	jle    8e92 <reed_sol_big_vandermonde_distribution_matrix+0x222>
    8cb6:	48 63 fb             	movslq %ebx,%rdi
    8cb9:	8d 43 fe             	lea    -0x2(%rbx),%eax
    8cbc:	44 89 64 24 44       	mov    %r12d,0x44(%rsp)
    8cc1:	48 8d 14 bd 00 00 00 	lea    0x0(,%rdi,4),%rdx
    8cc8:	00 
    8cc9:	48 83 c0 02          	add    $0x2,%rax
    8ccd:	48 89 7c 24 48       	mov    %rdi,0x48(%rsp)
    8cd2:	48 01 d1             	add    %rdx,%rcx
    8cd5:	48 89 54 24 08       	mov    %rdx,0x8(%rsp)
    8cda:	48 8d 57 01          	lea    0x1(%rdi),%rdx
    8cde:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    8ce3:	8d 43 ff             	lea    -0x1(%rbx),%eax
    8ce6:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    8ceb:	48 89 54 24 58       	mov    %rdx,0x58(%rsp)
    8cf0:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
    8cf5:	48 89 7c 24 50       	mov    %rdi,0x50(%rsp)
    8cfa:	48 c7 44 24 20 01 00 	movq   $0x1,0x20(%rsp)
    8d01:	00 00 
    8d03:	89 44 24 68          	mov    %eax,0x68(%rsp)
    8d07:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    8d0e:	00 00 
    8d10:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    8d15:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
    8d1a:	4c 8b 44 24 48       	mov    0x48(%rsp),%r8
    8d1f:	44 8b 4c 24 04       	mov    0x4(%rsp),%r9d
    8d24:	89 fe                	mov    %edi,%esi
    8d26:	89 fa                	mov    %edi,%edx
    8d28:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    8d2d:	0f 1f 00             	nopl   (%rax)
    8d30:	44 8b 1c 87          	mov    (%rdi,%rax,4),%r11d
    8d34:	89 c1                	mov    %eax,%ecx
    8d36:	45 85 db             	test   %r11d,%r11d
    8d39:	75 45                	jne    8d80 <reed_sol_big_vandermonde_distribution_matrix+0x110>
    8d3b:	83 c2 01             	add    $0x1,%edx
    8d3e:	4c 01 c0             	add    %r8,%rax
    8d41:	41 39 d1             	cmp    %edx,%r9d
    8d44:	7f ea                	jg     8d30 <reed_sol_big_vandermonde_distribution_matrix+0xc0>
    8d46:	44 8b 64 24 44       	mov    0x44(%rsp),%r12d
    8d4b:	8b 4c 24 04          	mov    0x4(%rsp),%ecx
    8d4f:	be 01 00 00 00       	mov    $0x1,%esi
    8d54:	31 c0                	xor    %eax,%eax
    8d56:	48 8b 3d e3 83 00 00 	mov    0x83e3(%rip),%rdi        # 11140 <stderr@@GLIBC_2.2.5>
    8d5d:	44 8b 44 24 6c       	mov    0x6c(%rsp),%r8d
    8d62:	48 8d 15 ff 1f 00 00 	lea    0x1fff(%rip),%rdx        # ad68 <prim_poly+0x88>
    8d69:	45 89 e1             	mov    %r12d,%r9d
    8d6c:	e8 ff 86 ff ff       	callq  1470 <__fprintf_chk@plt>
    8d71:	bf 01 00 00 00       	mov    $0x1,%edi
    8d76:	e8 d5 86 ff ff       	callq  1450 <exit@plt>
    8d7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8d80:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    8d85:	89 44 24 40          	mov    %eax,0x40(%rsp)
    8d89:	39 c2                	cmp    %eax,%edx
    8d8b:	0f 85 5f 02 00 00    	jne    8ff0 <reed_sol_big_vandermonde_distribution_matrix+0x380>
    8d91:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    8d96:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    8d9b:	8b 34 88             	mov    (%rax,%rcx,4),%esi
    8d9e:	83 fe 01             	cmp    $0x1,%esi
    8da1:	0f 85 89 02 00 00    	jne    9030 <reed_sol_big_vandermonde_distribution_matrix+0x3c0>
    8da7:	8b 44 24 68          	mov    0x68(%rsp),%eax
    8dab:	45 31 ff             	xor    %r15d,%r15d
    8dae:	4d 89 fe             	mov    %r15,%r14
    8db1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    8db6:	eb 1a                	jmp    8dd2 <reed_sol_big_vandermonde_distribution_matrix+0x162>
    8db8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    8dbf:	00 
    8dc0:	49 8d 46 01          	lea    0x1(%r14),%rax
    8dc4:	4c 3b 74 24 18       	cmp    0x18(%rsp),%r14
    8dc9:	0f 84 8a 00 00 00    	je     8e59 <reed_sol_big_vandermonde_distribution_matrix+0x1e9>
    8dcf:	49 89 c6             	mov    %rax,%r14
    8dd2:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    8dd7:	46 8b 2c b0          	mov    (%rax,%r14,4),%r13d
    8ddb:	44 39 74 24 40       	cmp    %r14d,0x40(%rsp)
    8de0:	74 de                	je     8dc0 <reed_sol_big_vandermonde_distribution_matrix+0x150>
    8de2:	45 85 ed             	test   %r13d,%r13d
    8de5:	74 d9                	je     8dc0 <reed_sol_big_vandermonde_distribution_matrix+0x150>
    8de7:	8b 54 24 04          	mov    0x4(%rsp),%edx
    8deb:	85 d2                	test   %edx,%edx
    8ded:	7e d1                	jle    8dc0 <reed_sol_big_vandermonde_distribution_matrix+0x150>
    8def:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    8df4:	4c 8b 7c 24 20       	mov    0x20(%rsp),%r15
    8df9:	45 31 e4             	xor    %r12d,%r12d
    8dfc:	4c 89 74 24 30       	mov    %r14,0x30(%rsp)
    8e01:	4d 29 f7             	sub    %r14,%r15
    8e04:	4a 8d 2c b0          	lea    (%rax,%r14,4),%rbp
    8e08:	49 89 ee             	mov    %rbp,%r14
    8e0b:	44 89 e5             	mov    %r12d,%ebp
    8e0e:	4d 89 fc             	mov    %r15,%r12
    8e11:	44 8b 7c 24 44       	mov    0x44(%rsp),%r15d
    8e16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    8e1d:	00 00 00 
    8e20:	43 8b 34 a6          	mov    (%r14,%r12,4),%esi
    8e24:	41 8b 1e             	mov    (%r14),%ebx
    8e27:	44 89 fa             	mov    %r15d,%edx
    8e2a:	44 89 ef             	mov    %r13d,%edi
    8e2d:	83 c5 01             	add    $0x1,%ebp
    8e30:	e8 db f6 ff ff       	callq  8510 <galois_single_multiply>
    8e35:	31 c3                	xor    %eax,%ebx
    8e37:	41 89 1e             	mov    %ebx,(%r14)
    8e3a:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    8e3f:	39 6c 24 04          	cmp    %ebp,0x4(%rsp)
    8e43:	75 db                	jne    8e20 <reed_sol_big_vandermonde_distribution_matrix+0x1b0>
    8e45:	4c 8b 74 24 30       	mov    0x30(%rsp),%r14
    8e4a:	49 8d 46 01          	lea    0x1(%r14),%rax
    8e4e:	4c 3b 74 24 18       	cmp    0x18(%rsp),%r14
    8e53:	0f 85 76 ff ff ff    	jne    8dcf <reed_sol_big_vandermonde_distribution_matrix+0x15f>
    8e59:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    8e5e:	48 83 44 24 20 01    	addq   $0x1,0x20(%rsp)
    8e64:	48 01 7c 24 10       	add    %rdi,0x10(%rsp)
    8e69:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    8e6e:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    8e73:	48 01 4c 24 50       	add    %rcx,0x50(%rsp)
    8e78:	48 01 7c 24 38       	add    %rdi,0x38(%rsp)
    8e7d:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
    8e82:	48 3b 44 24 60       	cmp    0x60(%rsp),%rax
    8e87:	0f 85 83 fe ff ff    	jne    8d10 <reed_sol_big_vandermonde_distribution_matrix+0xa0>
    8e8d:	44 8b 64 24 44       	mov    0x44(%rsp),%r12d
    8e92:	48 63 44 24 6c       	movslq 0x6c(%rsp),%rax
    8e97:	41 89 c6             	mov    %eax,%r14d
    8e9a:	44 0f af f0          	imul   %eax,%r14d
    8e9e:	85 c0                	test   %eax,%eax
    8ea0:	0f 8e 9a 00 00 00    	jle    8f40 <reed_sol_big_vandermonde_distribution_matrix+0x2d0>
    8ea6:	4d 63 f6             	movslq %r14d,%r14
    8ea9:	44 8d 68 ff          	lea    -0x1(%rax),%r13d
    8ead:	49 8d 5e 01          	lea    0x1(%r14),%rbx
    8eb1:	4c 89 f1             	mov    %r14,%rcx
    8eb4:	4d 8d 7c 1d 00       	lea    0x0(%r13,%rbx,1),%r15
    8eb9:	4c 8d 2c 85 00 00 00 	lea    0x0(,%rax,4),%r13
    8ec0:	00 
    8ec1:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
    8ec6:	eb 16                	jmp    8ede <reed_sol_big_vandermonde_distribution_matrix+0x26e>
    8ec8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    8ecf:	00 
    8ed0:	48 89 d9             	mov    %rbx,%rcx
    8ed3:	48 39 5c 24 08       	cmp    %rbx,0x8(%rsp)
    8ed8:	74 66                	je     8f40 <reed_sol_big_vandermonde_distribution_matrix+0x2d0>
    8eda:	48 83 c3 01          	add    $0x1,%rbx
    8ede:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    8ee3:	8b 34 88             	mov    (%rax,%rcx,4),%esi
    8ee6:	83 fe 01             	cmp    $0x1,%esi
    8ee9:	74 e5                	je     8ed0 <reed_sol_big_vandermonde_distribution_matrix+0x260>
    8eeb:	44 89 e2             	mov    %r12d,%edx
    8eee:	bf 01 00 00 00       	mov    $0x1,%edi
    8ef3:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    8ef8:	e8 33 f6 ff ff       	callq  8530 <galois_single_divide>
    8efd:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    8f02:	44 8b 7c 24 6c       	mov    0x6c(%rsp),%r15d
    8f07:	89 c5                	mov    %eax,%ebp
    8f09:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    8f0e:	4c 8d 34 88          	lea    (%rax,%rcx,4),%r14
    8f12:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    8f18:	41 8b 36             	mov    (%r14),%esi
    8f1b:	44 89 e2             	mov    %r12d,%edx
    8f1e:	89 ef                	mov    %ebp,%edi
    8f20:	41 83 c7 01          	add    $0x1,%r15d
    8f24:	e8 e7 f5 ff ff       	callq  8510 <galois_single_multiply>
    8f29:	41 89 06             	mov    %eax,(%r14)
    8f2c:	4d 01 ee             	add    %r13,%r14
    8f2f:	44 39 7c 24 04       	cmp    %r15d,0x4(%rsp)
    8f34:	75 e2                	jne    8f18 <reed_sol_big_vandermonde_distribution_matrix+0x2a8>
    8f36:	48 89 d9             	mov    %rbx,%rcx
    8f39:	48 39 5c 24 08       	cmp    %rbx,0x8(%rsp)
    8f3e:	75 9a                	jne    8eda <reed_sol_big_vandermonde_distribution_matrix+0x26a>
    8f40:	8b 44 24 6c          	mov    0x6c(%rsp),%eax
    8f44:	44 8d 68 01          	lea    0x1(%rax),%r13d
    8f48:	44 89 ed             	mov    %r13d,%ebp
    8f4b:	0f af e8             	imul   %eax,%ebp
    8f4e:	44 3b 6c 24 04       	cmp    0x4(%rsp),%r13d
    8f53:	0f 8d 40 01 00 00    	jge    9099 <reed_sol_big_vandermonde_distribution_matrix+0x429>
    8f59:	48 63 c8             	movslq %eax,%rcx
    8f5c:	48 63 ed             	movslq %ebp,%rbp
    8f5f:	83 e8 01             	sub    $0x1,%eax
    8f62:	4c 8d 3c 8d 00 00 00 	lea    0x0(,%rcx,4),%r15
    8f69:	00 
    8f6a:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
    8f6f:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
    8f74:	48 8d 44 05 01       	lea    0x1(%rbp,%rax,1),%rax
    8f79:	4c 89 7c 24 10       	mov    %r15,0x10(%rsp)
    8f7e:	48 8d 1c 81          	lea    (%rcx,%rax,4),%rbx
    8f82:	eb 1d                	jmp    8fa1 <reed_sol_big_vandermonde_distribution_matrix+0x331>
    8f84:	0f 1f 40 00          	nopl   0x0(%rax)
    8f88:	41 83 c5 01          	add    $0x1,%r13d
    8f8c:	48 03 6c 24 08       	add    0x8(%rsp),%rbp
    8f91:	48 03 5c 24 10       	add    0x10(%rsp),%rbx
    8f96:	44 39 6c 24 04       	cmp    %r13d,0x4(%rsp)
    8f9b:	0f 84 f8 00 00 00    	je     9099 <reed_sol_big_vandermonde_distribution_matrix+0x429>
    8fa1:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    8fa6:	8b 34 a8             	mov    (%rax,%rbp,4),%esi
    8fa9:	83 fe 01             	cmp    $0x1,%esi
    8fac:	74 da                	je     8f88 <reed_sol_big_vandermonde_distribution_matrix+0x318>
    8fae:	44 89 e2             	mov    %r12d,%edx
    8fb1:	bf 01 00 00 00       	mov    $0x1,%edi
    8fb6:	e8 75 f5 ff ff       	callq  8530 <galois_single_divide>
    8fbb:	41 89 c6             	mov    %eax,%r14d
    8fbe:	8b 44 24 6c          	mov    0x6c(%rsp),%eax
    8fc2:	85 c0                	test   %eax,%eax
    8fc4:	7e c2                	jle    8f88 <reed_sol_big_vandermonde_distribution_matrix+0x318>
    8fc6:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    8fcb:	4c 8d 3c a8          	lea    (%rax,%rbp,4),%r15
    8fcf:	90                   	nop
    8fd0:	41 8b 3f             	mov    (%r15),%edi
    8fd3:	44 89 e2             	mov    %r12d,%edx
    8fd6:	44 89 f6             	mov    %r14d,%esi
    8fd9:	49 83 c7 04          	add    $0x4,%r15
    8fdd:	e8 2e f5 ff ff       	callq  8510 <galois_single_multiply>
    8fe2:	41 89 47 fc          	mov    %eax,-0x4(%r15)
    8fe6:	49 39 df             	cmp    %rbx,%r15
    8fe9:	75 e5                	jne    8fd0 <reed_sol_big_vandermonde_distribution_matrix+0x360>
    8feb:	eb 9b                	jmp    8f88 <reed_sol_big_vandermonde_distribution_matrix+0x318>
    8fed:	0f 1f 00             	nopl   (%rax)
    8ff0:	8b 54 24 68          	mov    0x68(%rsp),%edx
    8ff4:	29 f1                	sub    %esi,%ecx
    8ff6:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    8ffb:	48 63 c9             	movslq %ecx,%rcx
    8ffe:	48 01 ca             	add    %rcx,%rdx
    9001:	48 8d 04 8f          	lea    (%rdi,%rcx,4),%rax
    9005:	48 8d 7c 97 04       	lea    0x4(%rdi,%rdx,4),%rdi
    900a:	48 8b 54 24 50       	mov    0x50(%rsp),%rdx
    900f:	48 29 ca             	sub    %rcx,%rdx
    9012:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    9018:	8b 08                	mov    (%rax),%ecx
    901a:	8b 34 90             	mov    (%rax,%rdx,4),%esi
    901d:	89 30                	mov    %esi,(%rax)
    901f:	89 0c 90             	mov    %ecx,(%rax,%rdx,4)
    9022:	48 83 c0 04          	add    $0x4,%rax
    9026:	48 39 f8             	cmp    %rdi,%rax
    9029:	75 ed                	jne    9018 <reed_sol_big_vandermonde_distribution_matrix+0x3a8>
    902b:	e9 61 fd ff ff       	jmpq   8d91 <reed_sol_big_vandermonde_distribution_matrix+0x121>
    9030:	44 8b 74 24 44       	mov    0x44(%rsp),%r14d
    9035:	bf 01 00 00 00       	mov    $0x1,%edi
    903a:	44 89 f2             	mov    %r14d,%edx
    903d:	e8 ee f4 ff ff       	callq  8530 <galois_single_divide>
    9042:	44 8b 6c 24 04       	mov    0x4(%rsp),%r13d
    9047:	89 c5                	mov    %eax,%ebp
    9049:	45 85 ed             	test   %r13d,%r13d
    904c:	0f 8e 55 fd ff ff    	jle    8da7 <reed_sol_big_vandermonde_distribution_matrix+0x137>
    9052:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9057:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    905c:	31 db                	xor    %ebx,%ebx
    905e:	4c 8b 64 24 08       	mov    0x8(%rsp),%r12
    9063:	4c 8d 3c 88          	lea    (%rax,%rcx,4),%r15
    9067:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    906e:	00 00 
    9070:	41 8b 37             	mov    (%r15),%esi
    9073:	44 89 f2             	mov    %r14d,%edx
    9076:	89 ef                	mov    %ebp,%edi
    9078:	83 c3 01             	add    $0x1,%ebx
    907b:	e8 90 f4 ff ff       	callq  8510 <galois_single_multiply>
    9080:	41 89 07             	mov    %eax,(%r15)
    9083:	4d 01 e7             	add    %r12,%r15
    9086:	41 39 dd             	cmp    %ebx,%r13d
    9089:	75 e5                	jne    9070 <reed_sol_big_vandermonde_distribution_matrix+0x400>
    908b:	e9 17 fd ff ff       	jmpq   8da7 <reed_sol_big_vandermonde_distribution_matrix+0x137>
    9090:	48 c7 44 24 28 00 00 	movq   $0x0,0x28(%rsp)
    9097:	00 00 
    9099:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    909e:	48 83 c4 78          	add    $0x78,%rsp
    90a2:	5b                   	pop    %rbx
    90a3:	5d                   	pop    %rbp
    90a4:	41 5c                	pop    %r12
    90a6:	41 5d                	pop    %r13
    90a8:	41 5e                	pop    %r14
    90aa:	41 5f                	pop    %r15
    90ac:	c3                   	retq   
    90ad:	0f 1f 00             	nopl   (%rax)

00000000000090b0 <reed_sol_vandermonde_coding_matrix>:
    90b0:	f3 0f 1e fa          	endbr64 
    90b4:	41 55                	push   %r13
    90b6:	41 54                	push   %r12
    90b8:	55                   	push   %rbp
    90b9:	89 f5                	mov    %esi,%ebp
    90bb:	53                   	push   %rbx
    90bc:	89 fb                	mov    %edi,%ebx
    90be:	01 f7                	add    %esi,%edi
    90c0:	89 de                	mov    %ebx,%esi
    90c2:	48 83 ec 08          	sub    $0x8,%rsp
    90c6:	e8 a5 fb ff ff       	callq  8c70 <reed_sol_big_vandermonde_distribution_matrix>
    90cb:	48 85 c0             	test   %rax,%rax
    90ce:	74 60                	je     9130 <reed_sol_vandermonde_coding_matrix+0x80>
    90d0:	0f af eb             	imul   %ebx,%ebp
    90d3:	49 89 c5             	mov    %rax,%r13
    90d6:	48 63 fd             	movslq %ebp,%rdi
    90d9:	48 c1 e7 02          	shl    $0x2,%rdi
    90dd:	e8 0e 83 ff ff       	callq  13f0 <malloc@plt>
    90e2:	49 89 c4             	mov    %rax,%r12
    90e5:	48 85 c0             	test   %rax,%rax
    90e8:	74 29                	je     9113 <reed_sol_vandermonde_coding_matrix+0x63>
    90ea:	0f af db             	imul   %ebx,%ebx
    90ed:	85 ed                	test   %ebp,%ebp
    90ef:	7e 22                	jle    9113 <reed_sol_vandermonde_coding_matrix+0x63>
    90f1:	48 63 db             	movslq %ebx,%rbx
    90f4:	8d 75 ff             	lea    -0x1(%rbp),%esi
    90f7:	31 d2                	xor    %edx,%edx
    90f9:	49 8d 44 9d 00       	lea    0x0(%r13,%rbx,4),%rax
    90fe:	66 90                	xchg   %ax,%ax
    9100:	8b 0c 90             	mov    (%rax,%rdx,4),%ecx
    9103:	41 89 0c 94          	mov    %ecx,(%r12,%rdx,4)
    9107:	48 89 d1             	mov    %rdx,%rcx
    910a:	48 83 c2 01          	add    $0x1,%rdx
    910e:	48 39 ce             	cmp    %rcx,%rsi
    9111:	75 ed                	jne    9100 <reed_sol_vandermonde_coding_matrix+0x50>
    9113:	4c 89 ef             	mov    %r13,%rdi
    9116:	e8 55 81 ff ff       	callq  1270 <free@plt>
    911b:	48 83 c4 08          	add    $0x8,%rsp
    911f:	4c 89 e0             	mov    %r12,%rax
    9122:	5b                   	pop    %rbx
    9123:	5d                   	pop    %rbp
    9124:	41 5c                	pop    %r12
    9126:	41 5d                	pop    %r13
    9128:	c3                   	retq   
    9129:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    9130:	48 83 c4 08          	add    $0x8,%rsp
    9134:	45 31 e4             	xor    %r12d,%r12d
    9137:	5b                   	pop    %rbx
    9138:	4c 89 e0             	mov    %r12,%rax
    913b:	5d                   	pop    %rbp
    913c:	41 5c                	pop    %r12
    913e:	41 5d                	pop    %r13
    9140:	c3                   	retq   
    9141:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    9148:	00 00 00 
    914b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000009150 <cauchy_n_ones>:
    9150:	f3 0f 1e fa          	endbr64 
    9154:	41 57                	push   %r15
    9156:	41 56                	push   %r14
    9158:	41 be 01 00 00 00    	mov    $0x1,%r14d
    915e:	41 55                	push   %r13
    9160:	4c 8d 2d 39 7f 00 00 	lea    0x7f39(%rip),%r13        # 110a0 <PPs>
    9167:	41 54                	push   %r12
    9169:	4c 63 e6             	movslq %esi,%r12
    916c:	55                   	push   %rbp
    916d:	41 8d 4c 24 ff       	lea    -0x1(%r12),%ecx
    9172:	4c 89 e5             	mov    %r12,%rbp
    9175:	53                   	push   %rbx
    9176:	41 d3 e6             	shl    %cl,%r14d
    9179:	89 fb                	mov    %edi,%ebx
    917b:	48 83 ec 18          	sub    $0x18,%rsp
    917f:	43 83 7c a5 00 ff    	cmpl   $0xffffffff,0x0(%r13,%r12,4)
    9185:	0f 84 d5 00 00 00    	je     9260 <cauchy_n_ones+0x110>
    918b:	85 ed                	test   %ebp,%ebp
    918d:	0f 8e 37 01 00 00    	jle    92ca <cauchy_n_ones+0x17a>
    9193:	31 c9                	xor    %ecx,%ecx
    9195:	31 f6                	xor    %esi,%esi
    9197:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    919e:	00 00 
    91a0:	89 d8                	mov    %ebx,%eax
    91a2:	d3 f8                	sar    %cl,%eax
    91a4:	83 e0 01             	and    $0x1,%eax
    91a7:	83 f8 01             	cmp    $0x1,%eax
    91aa:	83 de ff             	sbb    $0xffffffff,%esi
    91ad:	83 c1 01             	add    $0x1,%ecx
    91b0:	39 cd                	cmp    %ecx,%ebp
    91b2:	75 ec                	jne    91a0 <cauchy_n_ones+0x50>
    91b4:	83 fd 01             	cmp    $0x1,%ebp
    91b7:	0f 8e 19 01 00 00    	jle    92d6 <cauchy_n_ones+0x186>
    91bd:	4d 89 e1             	mov    %r12,%r9
    91c0:	48 8d 05 99 85 00 00 	lea    0x8599(%rip),%rax        # 11760 <ONEs>
    91c7:	41 89 f0             	mov    %esi,%r8d
    91ca:	b9 01 00 00 00       	mov    $0x1,%ecx
    91cf:	49 c1 e1 05          	shl    $0x5,%r9
    91d3:	4c 8d 1d a6 96 00 00 	lea    0x96a6(%rip),%r11        # 12880 <NOs>
    91da:	4c 8d 3d 83 85 00 00 	lea    0x8583(%rip),%r15        # 11764 <ONEs+0x4>
    91e1:	4d 01 e1             	add    %r12,%r9
    91e4:	4a 8d 04 88          	lea    (%rax,%r9,4),%rax
    91e8:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    91ed:	eb 0d                	jmp    91fc <cauchy_n_ones+0xac>
    91ef:	90                   	nop
    91f0:	01 db                	add    %ebx,%ebx
    91f2:	83 c1 01             	add    $0x1,%ecx
    91f5:	41 01 f0             	add    %esi,%r8d
    91f8:	39 cd                	cmp    %ecx,%ebp
    91fa:	74 4d                	je     9249 <cauchy_n_ones+0xf9>
    91fc:	41 85 de             	test   %ebx,%r14d
    91ff:	74 ef                	je     91f0 <cauchy_n_ones+0xa0>
    9201:	43 8b 04 a3          	mov    (%r11,%r12,4),%eax
    9205:	44 31 f3             	xor    %r14d,%ebx
    9208:	83 ee 01             	sub    $0x1,%esi
    920b:	01 db                	add    %ebx,%ebx
    920d:	43 33 5c a5 00       	xor    0x0(%r13,%r12,4),%ebx
    9212:	85 c0                	test   %eax,%eax
    9214:	7e dc                	jle    91f2 <cauchy_n_ones+0xa2>
    9216:	83 e8 01             	sub    $0x1,%eax
    9219:	4c 01 c8             	add    %r9,%rax
    921c:	49 8d 3c 87          	lea    (%r15,%rax,4),%rdi
    9220:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    9225:	0f 1f 00             	nopl   (%rax)
    9228:	8b 10                	mov    (%rax),%edx
    922a:	21 da                	and    %ebx,%edx
    922c:	83 fa 01             	cmp    $0x1,%edx
    922f:	19 d2                	sbb    %edx,%edx
    9231:	48 83 c0 04          	add    $0x4,%rax
    9235:	83 ca 01             	or     $0x1,%edx
    9238:	01 d6                	add    %edx,%esi
    923a:	48 39 f8             	cmp    %rdi,%rax
    923d:	75 e9                	jne    9228 <cauchy_n_ones+0xd8>
    923f:	83 c1 01             	add    $0x1,%ecx
    9242:	41 01 f0             	add    %esi,%r8d
    9245:	39 cd                	cmp    %ecx,%ebp
    9247:	75 b3                	jne    91fc <cauchy_n_ones+0xac>
    9249:	48 83 c4 18          	add    $0x18,%rsp
    924d:	44 89 c0             	mov    %r8d,%eax
    9250:	5b                   	pop    %rbx
    9251:	5d                   	pop    %rbp
    9252:	41 5c                	pop    %r12
    9254:	41 5d                	pop    %r13
    9256:	41 5e                	pop    %r14
    9258:	41 5f                	pop    %r15
    925a:	c3                   	retq   
    925b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    9260:	44 89 e2             	mov    %r12d,%edx
    9263:	be 02 00 00 00       	mov    $0x2,%esi
    9268:	44 89 f7             	mov    %r14d,%edi
    926b:	e8 a0 f2 ff ff       	callq  8510 <galois_single_multiply>
    9270:	43 89 44 a5 00       	mov    %eax,0x0(%r13,%r12,4)
    9275:	45 85 e4             	test   %r12d,%r12d
    9278:	7e 58                	jle    92d2 <cauchy_n_ones+0x182>
    927a:	4c 89 e7             	mov    %r12,%rdi
    927d:	31 c9                	xor    %ecx,%ecx
    927f:	31 f6                	xor    %esi,%esi
    9281:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    9287:	48 c1 e7 05          	shl    $0x5,%rdi
    928b:	4c 8d 0d ce 84 00 00 	lea    0x84ce(%rip),%r9        # 11760 <ONEs>
    9292:	4c 01 e7             	add    %r12,%rdi
    9295:	0f 1f 00             	nopl   (%rax)
    9298:	0f a3 c8             	bt     %ecx,%eax
    929b:	73 13                	jae    92b0 <cauchy_n_ones+0x160>
    929d:	48 63 d6             	movslq %esi,%rdx
    92a0:	45 89 c3             	mov    %r8d,%r11d
    92a3:	83 c6 01             	add    $0x1,%esi
    92a6:	48 01 fa             	add    %rdi,%rdx
    92a9:	41 d3 e3             	shl    %cl,%r11d
    92ac:	45 89 1c 91          	mov    %r11d,(%r9,%rdx,4)
    92b0:	83 c1 01             	add    $0x1,%ecx
    92b3:	39 cd                	cmp    %ecx,%ebp
    92b5:	75 e1                	jne    9298 <cauchy_n_ones+0x148>
    92b7:	48 8d 05 c2 95 00 00 	lea    0x95c2(%rip),%rax        # 12880 <NOs>
    92be:	42 89 34 a0          	mov    %esi,(%rax,%r12,4)
    92c2:	85 ed                	test   %ebp,%ebp
    92c4:	0f 8f c9 fe ff ff    	jg     9193 <cauchy_n_ones+0x43>
    92ca:	45 31 c0             	xor    %r8d,%r8d
    92cd:	e9 77 ff ff ff       	jmpq   9249 <cauchy_n_ones+0xf9>
    92d2:	31 f6                	xor    %esi,%esi
    92d4:	eb e1                	jmp    92b7 <cauchy_n_ones+0x167>
    92d6:	41 89 f0             	mov    %esi,%r8d
    92d9:	e9 6b ff ff ff       	jmpq   9249 <cauchy_n_ones+0xf9>
    92de:	66 90                	xchg   %ax,%ax

00000000000092e0 <cauchy_original_coding_matrix>:
    92e0:	f3 0f 1e fa          	endbr64 
    92e4:	41 57                	push   %r15
    92e6:	41 89 d7             	mov    %edx,%r15d
    92e9:	41 56                	push   %r14
    92eb:	41 55                	push   %r13
    92ed:	41 54                	push   %r12
    92ef:	55                   	push   %rbp
    92f0:	53                   	push   %rbx
    92f1:	48 83 ec 18          	sub    $0x18,%rsp
    92f5:	89 3c 24             	mov    %edi,(%rsp)
    92f8:	89 74 24 04          	mov    %esi,0x4(%rsp)
    92fc:	83 fa 1e             	cmp    $0x1e,%edx
    92ff:	7f 16                	jg     9317 <cauchy_original_coding_matrix+0x37>
    9301:	89 fa                	mov    %edi,%edx
    9303:	b8 01 00 00 00       	mov    $0x1,%eax
    9308:	44 89 f9             	mov    %r15d,%ecx
    930b:	01 f2                	add    %esi,%edx
    930d:	d3 e0                	shl    %cl,%eax
    930f:	39 c2                	cmp    %eax,%edx
    9311:	0f 8f 9d 00 00 00    	jg     93b4 <cauchy_original_coding_matrix+0xd4>
    9317:	44 8b 2c 24          	mov    (%rsp),%r13d
    931b:	44 8b 74 24 04       	mov    0x4(%rsp),%r14d
    9320:	44 89 ef             	mov    %r13d,%edi
    9323:	41 0f af fe          	imul   %r14d,%edi
    9327:	48 63 ff             	movslq %edi,%rdi
    932a:	48 c1 e7 02          	shl    $0x2,%rdi
    932e:	e8 bd 80 ff ff       	callq  13f0 <malloc@plt>
    9333:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    9338:	48 85 c0             	test   %rax,%rax
    933b:	74 77                	je     93b4 <cauchy_original_coding_matrix+0xd4>
    933d:	44 89 f5             	mov    %r14d,%ebp
    9340:	45 31 e4             	xor    %r12d,%r12d
    9343:	31 db                	xor    %ebx,%ebx
    9345:	44 01 ed             	add    %r13d,%ebp
    9348:	45 85 f6             	test   %r14d,%r14d
    934b:	7e 53                	jle    93a0 <cauchy_original_coding_matrix+0xc0>
    934d:	0f 1f 00             	nopl   (%rax)
    9350:	8b 04 24             	mov    (%rsp),%eax
    9353:	85 c0                	test   %eax,%eax
    9355:	7e 40                	jle    9397 <cauchy_original_coding_matrix+0xb7>
    9357:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    935c:	49 63 c4             	movslq %r12d,%rax
    935f:	44 8b 74 24 04       	mov    0x4(%rsp),%r14d
    9364:	4c 8d 2c 81          	lea    (%rcx,%rax,4),%r13
    9368:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    936f:	00 
    9370:	44 89 f6             	mov    %r14d,%esi
    9373:	44 89 fa             	mov    %r15d,%edx
    9376:	bf 01 00 00 00       	mov    $0x1,%edi
    937b:	41 83 c6 01          	add    $0x1,%r14d
    937f:	31 de                	xor    %ebx,%esi
    9381:	49 83 c5 04          	add    $0x4,%r13
    9385:	e8 a6 f1 ff ff       	callq  8530 <galois_single_divide>
    938a:	41 89 45 fc          	mov    %eax,-0x4(%r13)
    938e:	41 39 ee             	cmp    %ebp,%r14d
    9391:	75 dd                	jne    9370 <cauchy_original_coding_matrix+0x90>
    9393:	44 03 24 24          	add    (%rsp),%r12d
    9397:	83 c3 01             	add    $0x1,%ebx
    939a:	39 5c 24 04          	cmp    %ebx,0x4(%rsp)
    939e:	75 b0                	jne    9350 <cauchy_original_coding_matrix+0x70>
    93a0:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    93a5:	48 83 c4 18          	add    $0x18,%rsp
    93a9:	5b                   	pop    %rbx
    93aa:	5d                   	pop    %rbp
    93ab:	41 5c                	pop    %r12
    93ad:	41 5d                	pop    %r13
    93af:	41 5e                	pop    %r14
    93b1:	41 5f                	pop    %r15
    93b3:	c3                   	retq   
    93b4:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    93bb:	00 00 
    93bd:	eb e1                	jmp    93a0 <cauchy_original_coding_matrix+0xc0>
    93bf:	90                   	nop

00000000000093c0 <cauchy_xy_coding_matrix>:
    93c0:	f3 0f 1e fa          	endbr64 
    93c4:	41 57                	push   %r15
    93c6:	49 89 cf             	mov    %rcx,%r15
    93c9:	41 56                	push   %r14
    93cb:	41 89 fe             	mov    %edi,%r14d
    93ce:	41 55                	push   %r13
    93d0:	41 54                	push   %r12
    93d2:	41 89 f4             	mov    %esi,%r12d
    93d5:	55                   	push   %rbp
    93d6:	4c 89 c5             	mov    %r8,%rbp
    93d9:	53                   	push   %rbx
    93da:	89 d3                	mov    %edx,%ebx
    93dc:	48 83 ec 28          	sub    $0x28,%rsp
    93e0:	89 7c 24 04          	mov    %edi,0x4(%rsp)
    93e4:	0f af fe             	imul   %esi,%edi
    93e7:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
    93ec:	48 63 ff             	movslq %edi,%rdi
    93ef:	48 c1 e7 02          	shl    $0x2,%rdi
    93f3:	e8 f8 7f ff ff       	callq  13f0 <malloc@plt>
    93f8:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    93fd:	48 85 c0             	test   %rax,%rax
    9400:	74 79                	je     947b <cauchy_xy_coding_matrix+0xbb>
    9402:	45 85 e4             	test   %r12d,%r12d
    9405:	7e 74                	jle    947b <cauchy_xy_coding_matrix+0xbb>
    9407:	41 8d 44 24 ff       	lea    -0x1(%r12),%eax
    940c:	45 31 e4             	xor    %r12d,%r12d
    940f:	49 8d 44 87 04       	lea    0x4(%r15,%rax,4),%rax
    9414:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    9419:	41 8d 46 ff          	lea    -0x1(%r14),%eax
    941d:	48 8d 6c 85 04       	lea    0x4(%rbp,%rax,4),%rbp
    9422:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    9428:	8b 44 24 04          	mov    0x4(%rsp),%eax
    942c:	85 c0                	test   %eax,%eax
    942e:	7e 40                	jle    9470 <cauchy_xy_coding_matrix+0xb0>
    9430:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    9435:	49 63 c4             	movslq %r12d,%rax
    9438:	4c 8b 74 24 18       	mov    0x18(%rsp),%r14
    943d:	4c 8d 2c 81          	lea    (%rcx,%rax,4),%r13
    9441:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    9448:	41 8b 37             	mov    (%r15),%esi
    944b:	89 da                	mov    %ebx,%edx
    944d:	41 33 36             	xor    (%r14),%esi
    9450:	bf 01 00 00 00       	mov    $0x1,%edi
    9455:	e8 d6 f0 ff ff       	callq  8530 <galois_single_divide>
    945a:	49 83 c6 04          	add    $0x4,%r14
    945e:	49 83 c5 04          	add    $0x4,%r13
    9462:	41 89 45 fc          	mov    %eax,-0x4(%r13)
    9466:	49 39 ee             	cmp    %rbp,%r14
    9469:	75 dd                	jne    9448 <cauchy_xy_coding_matrix+0x88>
    946b:	44 03 64 24 04       	add    0x4(%rsp),%r12d
    9470:	49 83 c7 04          	add    $0x4,%r15
    9474:	4c 3b 7c 24 10       	cmp    0x10(%rsp),%r15
    9479:	75 ad                	jne    9428 <cauchy_xy_coding_matrix+0x68>
    947b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    9480:	48 83 c4 28          	add    $0x28,%rsp
    9484:	5b                   	pop    %rbx
    9485:	5d                   	pop    %rbp
    9486:	41 5c                	pop    %r12
    9488:	41 5d                	pop    %r13
    948a:	41 5e                	pop    %r14
    948c:	41 5f                	pop    %r15
    948e:	c3                   	retq   
    948f:	90                   	nop

0000000000009490 <cauchy_improve_coding_matrix>:
    9490:	f3 0f 1e fa          	endbr64 
    9494:	41 57                	push   %r15
    9496:	41 56                	push   %r14
    9498:	41 55                	push   %r13
    949a:	41 89 d5             	mov    %edx,%r13d
    949d:	41 54                	push   %r12
    949f:	55                   	push   %rbp
    94a0:	53                   	push   %rbx
    94a1:	48 83 ec 48          	sub    $0x48,%rsp
    94a5:	89 7c 24 30          	mov    %edi,0x30(%rsp)
    94a9:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
    94ad:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
    94b2:	85 ff                	test   %edi,%edi
    94b4:	0f 8e 7e 00 00 00    	jle    9538 <cauchy_improve_coding_matrix+0xa8>
    94ba:	8d 47 ff             	lea    -0x1(%rdi),%eax
    94bd:	4c 63 e7             	movslq %edi,%r12
    94c0:	31 db                	xor    %ebx,%ebx
    94c2:	48 89 04 24          	mov    %rax,(%rsp)
    94c6:	49 c1 e4 02          	shl    $0x2,%r12
    94ca:	eb 11                	jmp    94dd <cauchy_improve_coding_matrix+0x4d>
    94cc:	0f 1f 40 00          	nopl   0x0(%rax)
    94d0:	48 8d 43 01          	lea    0x1(%rbx),%rax
    94d4:	48 3b 1c 24          	cmp    (%rsp),%rbx
    94d8:	74 5e                	je     9538 <cauchy_improve_coding_matrix+0xa8>
    94da:	48 89 c3             	mov    %rax,%rbx
    94dd:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    94e2:	8b 34 98             	mov    (%rax,%rbx,4),%esi
    94e5:	83 fe 01             	cmp    $0x1,%esi
    94e8:	74 e6                	je     94d0 <cauchy_improve_coding_matrix+0x40>
    94ea:	44 89 ea             	mov    %r13d,%edx
    94ed:	bf 01 00 00 00       	mov    $0x1,%edi
    94f2:	e8 39 f0 ff ff       	callq  8530 <galois_single_divide>
    94f7:	8b 54 24 1c          	mov    0x1c(%rsp),%edx
    94fb:	89 c5                	mov    %eax,%ebp
    94fd:	85 d2                	test   %edx,%edx
    94ff:	7e cf                	jle    94d0 <cauchy_improve_coding_matrix+0x40>
    9501:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    9506:	45 31 ff             	xor    %r15d,%r15d
    9509:	4c 8d 34 98          	lea    (%rax,%rbx,4),%r14
    950d:	0f 1f 00             	nopl   (%rax)
    9510:	41 8b 3e             	mov    (%r14),%edi
    9513:	44 89 ea             	mov    %r13d,%edx
    9516:	89 ee                	mov    %ebp,%esi
    9518:	41 83 c7 01          	add    $0x1,%r15d
    951c:	e8 ef ef ff ff       	callq  8510 <galois_single_multiply>
    9521:	41 89 06             	mov    %eax,(%r14)
    9524:	4d 01 e6             	add    %r12,%r14
    9527:	44 39 7c 24 1c       	cmp    %r15d,0x1c(%rsp)
    952c:	75 e2                	jne    9510 <cauchy_improve_coding_matrix+0x80>
    952e:	48 8d 43 01          	lea    0x1(%rbx),%rax
    9532:	48 3b 1c 24          	cmp    (%rsp),%rbx
    9536:	75 a2                	jne    94da <cauchy_improve_coding_matrix+0x4a>
    9538:	83 7c 24 1c 01       	cmpl   $0x1,0x1c(%rsp)
    953d:	0f 8e 3b 01 00 00    	jle    967e <cauchy_improve_coding_matrix+0x1ee>
    9543:	48 63 44 24 30       	movslq 0x30(%rsp),%rax
    9548:	c7 44 24 20 01 00 00 	movl   $0x1,0x20(%rsp)
    954f:	00 
    9550:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    9557:	00 
    9558:	48 89 c1             	mov    %rax,%rcx
    955b:	48 89 d3             	mov    %rdx,%rbx
    955e:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
    9563:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    9568:	89 4c 24 34          	mov    %ecx,0x34(%rsp)
    956c:	48 89 d6             	mov    %rdx,%rsi
    956f:	48 01 de             	add    %rbx,%rsi
    9572:	48 89 34 24          	mov    %rsi,(%rsp)
    9576:	8d 70 ff             	lea    -0x1(%rax),%esi
    9579:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
    957e:	48 01 f0             	add    %rsi,%rax
    9581:	4c 8d 7c 82 04       	lea    0x4(%rdx,%rax,4),%r15
    9586:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    958d:	00 00 00 
    9590:	8b 44 24 30          	mov    0x30(%rsp),%eax
    9594:	85 c0                	test   %eax,%eax
    9596:	0f 8e bb 00 00 00    	jle    9657 <cauchy_improve_coding_matrix+0x1c7>
    959c:	48 8b 1c 24          	mov    (%rsp),%rbx
    95a0:	31 ed                	xor    %ebp,%ebp
    95a2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    95a8:	8b 3b                	mov    (%rbx),%edi
    95aa:	44 89 ee             	mov    %r13d,%esi
    95ad:	48 83 c3 04          	add    $0x4,%rbx
    95b1:	e8 9a fb ff ff       	callq  9150 <cauchy_n_ones>
    95b6:	01 c5                	add    %eax,%ebp
    95b8:	4c 39 fb             	cmp    %r15,%rbx
    95bb:	75 eb                	jne    95a8 <cauchy_improve_coding_matrix+0x118>
    95bd:	89 6c 24 18          	mov    %ebp,0x18(%rsp)
    95c1:	31 db                	xor    %ebx,%ebx
    95c3:	c7 44 24 24 ff ff ff 	movl   $0xffffffff,0x24(%rsp)
    95ca:	ff 
    95cb:	eb 11                	jmp    95de <cauchy_improve_coding_matrix+0x14e>
    95cd:	0f 1f 00             	nopl   (%rax)
    95d0:	48 8d 43 01          	lea    0x1(%rbx),%rax
    95d4:	48 3b 5c 24 10       	cmp    0x10(%rsp),%rbx
    95d9:	74 75                	je     9650 <cauchy_improve_coding_matrix+0x1c0>
    95db:	48 89 c3             	mov    %rax,%rbx
    95de:	48 8b 04 24          	mov    (%rsp),%rax
    95e2:	89 5c 24 0c          	mov    %ebx,0xc(%rsp)
    95e6:	8b 34 98             	mov    (%rax,%rbx,4),%esi
    95e9:	83 fe 01             	cmp    $0x1,%esi
    95ec:	74 e2                	je     95d0 <cauchy_improve_coding_matrix+0x140>
    95ee:	44 89 ea             	mov    %r13d,%edx
    95f1:	bf 01 00 00 00       	mov    $0x1,%edi
    95f6:	31 ed                	xor    %ebp,%ebp
    95f8:	e8 33 ef ff ff       	callq  8530 <galois_single_divide>
    95fd:	4c 8b 34 24          	mov    (%rsp),%r14
    9601:	41 89 c4             	mov    %eax,%r12d
    9604:	0f 1f 40 00          	nopl   0x0(%rax)
    9608:	41 8b 3e             	mov    (%r14),%edi
    960b:	44 89 ea             	mov    %r13d,%edx
    960e:	44 89 e6             	mov    %r12d,%esi
    9611:	49 83 c6 04          	add    $0x4,%r14
    9615:	e8 f6 ee ff ff       	callq  8510 <galois_single_multiply>
    961a:	44 89 ee             	mov    %r13d,%esi
    961d:	89 c7                	mov    %eax,%edi
    961f:	e8 2c fb ff ff       	callq  9150 <cauchy_n_ones>
    9624:	01 c5                	add    %eax,%ebp
    9626:	4d 39 fe             	cmp    %r15,%r14
    9629:	75 dd                	jne    9608 <cauchy_improve_coding_matrix+0x178>
    962b:	3b 6c 24 18          	cmp    0x18(%rsp),%ebp
    962f:	7d 9f                	jge    95d0 <cauchy_improve_coding_matrix+0x140>
    9631:	8b 44 24 0c          	mov    0xc(%rsp),%eax
    9635:	89 6c 24 18          	mov    %ebp,0x18(%rsp)
    9639:	89 44 24 24          	mov    %eax,0x24(%rsp)
    963d:	48 8d 43 01          	lea    0x1(%rbx),%rax
    9641:	48 3b 5c 24 10       	cmp    0x10(%rsp),%rbx
    9646:	75 93                	jne    95db <cauchy_improve_coding_matrix+0x14b>
    9648:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    964f:	00 
    9650:	83 7c 24 24 ff       	cmpl   $0xffffffff,0x24(%rsp)
    9655:	75 39                	jne    9690 <cauchy_improve_coding_matrix+0x200>
    9657:	48 8b 54 24 38       	mov    0x38(%rsp),%rdx
    965c:	83 44 24 20 01       	addl   $0x1,0x20(%rsp)
    9661:	8b 74 24 30          	mov    0x30(%rsp),%esi
    9665:	48 01 14 24          	add    %rdx,(%rsp)
    9669:	01 74 24 34          	add    %esi,0x34(%rsp)
    966d:	8b 44 24 20          	mov    0x20(%rsp),%eax
    9671:	49 01 d7             	add    %rdx,%r15
    9674:	39 44 24 1c          	cmp    %eax,0x1c(%rsp)
    9678:	0f 85 12 ff ff ff    	jne    9590 <cauchy_improve_coding_matrix+0x100>
    967e:	48 83 c4 48          	add    $0x48,%rsp
    9682:	5b                   	pop    %rbx
    9683:	5d                   	pop    %rbp
    9684:	41 5c                	pop    %r12
    9686:	41 5d                	pop    %r13
    9688:	41 5e                	pop    %r14
    968a:	41 5f                	pop    %r15
    968c:	c3                   	retq   
    968d:	0f 1f 00             	nopl   (%rax)
    9690:	8b 44 24 24          	mov    0x24(%rsp),%eax
    9694:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
    9699:	44 89 ea             	mov    %r13d,%edx
    969c:	bf 01 00 00 00       	mov    $0x1,%edi
    96a1:	03 44 24 34          	add    0x34(%rsp),%eax
    96a5:	48 98                	cltq   
    96a7:	8b 34 81             	mov    (%rcx,%rax,4),%esi
    96aa:	e8 81 ee ff ff       	callq  8530 <galois_single_divide>
    96af:	48 8b 2c 24          	mov    (%rsp),%rbp
    96b3:	89 c3                	mov    %eax,%ebx
    96b5:	0f 1f 00             	nopl   (%rax)
    96b8:	8b 7d 00             	mov    0x0(%rbp),%edi
    96bb:	44 89 ea             	mov    %r13d,%edx
    96be:	89 de                	mov    %ebx,%esi
    96c0:	48 83 c5 04          	add    $0x4,%rbp
    96c4:	e8 47 ee ff ff       	callq  8510 <galois_single_multiply>
    96c9:	89 45 fc             	mov    %eax,-0x4(%rbp)
    96cc:	4c 39 fd             	cmp    %r15,%rbp
    96cf:	75 e7                	jne    96b8 <cauchy_improve_coding_matrix+0x228>
    96d1:	eb 84                	jmp    9657 <cauchy_improve_coding_matrix+0x1c7>
    96d3:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    96da:	00 00 00 00 
    96de:	66 90                	xchg   %ax,%ax

00000000000096e0 <cauchy_good_general_coding_matrix>:
    96e0:	f3 0f 1e fa          	endbr64 
    96e4:	41 56                	push   %r14
    96e6:	41 89 fe             	mov    %edi,%r14d
    96e9:	41 55                	push   %r13
    96eb:	41 89 d5             	mov    %edx,%r13d
    96ee:	41 54                	push   %r12
    96f0:	55                   	push   %rbp
    96f1:	89 f5                	mov    %esi,%ebp
    96f3:	53                   	push   %rbx
    96f4:	83 fe 02             	cmp    $0x2,%esi
    96f7:	75 0f                	jne    9708 <cauchy_good_general_coding_matrix+0x28>
    96f9:	48 63 da             	movslq %edx,%rbx
    96fc:	48 8d 05 bd 16 00 00 	lea    0x16bd(%rip),%rax        # adc0 <cbest_max_k>
    9703:	39 3c 98             	cmp    %edi,(%rax,%rbx,4)
    9706:	7d 38                	jge    9740 <cauchy_good_general_coding_matrix+0x60>
    9708:	44 89 ea             	mov    %r13d,%edx
    970b:	89 ee                	mov    %ebp,%esi
    970d:	44 89 f7             	mov    %r14d,%edi
    9710:	e8 cb fb ff ff       	callq  92e0 <cauchy_original_coding_matrix>
    9715:	49 89 c4             	mov    %rax,%r12
    9718:	48 85 c0             	test   %rax,%rax
    971b:	74 10                	je     972d <cauchy_good_general_coding_matrix+0x4d>
    971d:	48 89 c1             	mov    %rax,%rcx
    9720:	44 89 ea             	mov    %r13d,%edx
    9723:	89 ee                	mov    %ebp,%esi
    9725:	44 89 f7             	mov    %r14d,%edi
    9728:	e8 63 fd ff ff       	callq  9490 <cauchy_improve_coding_matrix>
    972d:	5b                   	pop    %rbx
    972e:	4c 89 e0             	mov    %r12,%rax
    9731:	5d                   	pop    %rbp
    9732:	41 5c                	pop    %r12
    9734:	41 5d                	pop    %r13
    9736:	41 5e                	pop    %r14
    9738:	c3                   	retq   
    9739:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    9740:	8d 3c 3f             	lea    (%rdi,%rdi,1),%edi
    9743:	48 63 ff             	movslq %edi,%rdi
    9746:	48 c1 e7 02          	shl    $0x2,%rdi
    974a:	e8 a1 7c ff ff       	callq  13f0 <malloc@plt>
    974f:	49 89 c4             	mov    %rax,%r12
    9752:	48 85 c0             	test   %rax,%rax
    9755:	74 d6                	je     972d <cauchy_good_general_coding_matrix+0x4d>
    9757:	8b 05 eb 7f 00 00    	mov    0x7feb(%rip),%eax        # 11748 <cbest_init>
    975d:	85 c0                	test   %eax,%eax
    975f:	0f 85 93 01 00 00    	jne    98f8 <cauchy_good_general_coding_matrix+0x218>
    9765:	48 8d 05 14 79 00 00 	lea    0x7914(%rip),%rax        # 11080 <cbest_2>
    976c:	c7 05 d2 7f 00 00 01 	movl   $0x1,0x7fd2(%rip)        # 11748 <cbest_init>
    9773:	00 00 00 
    9776:	48 89 05 d3 7e 00 00 	mov    %rax,0x7ed3(%rip)        # 11650 <cbest_all+0x10>
    977d:	48 8d 05 dc 78 00 00 	lea    0x78dc(%rip),%rax        # 11060 <cbest_3>
    9784:	48 89 05 cd 7e 00 00 	mov    %rax,0x7ecd(%rip)        # 11658 <cbest_all+0x18>
    978b:	48 8d 05 8e 78 00 00 	lea    0x788e(%rip),%rax        # 11020 <cbest_4>
    9792:	48 89 05 c7 7e 00 00 	mov    %rax,0x7ec7(%rip)        # 11660 <cbest_all+0x20>
    9799:	48 8d 05 00 78 00 00 	lea    0x7800(%rip),%rax        # 10fa0 <cbest_5>
    97a0:	48 89 05 c1 7e 00 00 	mov    %rax,0x7ec1(%rip)        # 11668 <cbest_all+0x28>
    97a7:	48 8d 05 f2 76 00 00 	lea    0x76f2(%rip),%rax        # 10ea0 <cbest_6>
    97ae:	48 89 05 bb 7e 00 00 	mov    %rax,0x7ebb(%rip)        # 11670 <cbest_all+0x30>
    97b5:	48 8d 05 e4 74 00 00 	lea    0x74e4(%rip),%rax        # 10ca0 <cbest_7>
    97bc:	48 89 05 b5 7e 00 00 	mov    %rax,0x7eb5(%rip)        # 11678 <cbest_all+0x38>
    97c3:	48 8d 05 d6 70 00 00 	lea    0x70d6(%rip),%rax        # 108a0 <cbest_8>
    97ca:	48 89 05 af 7e 00 00 	mov    %rax,0x7eaf(%rip)        # 11680 <cbest_all+0x40>
    97d1:	48 8d 05 c8 68 00 00 	lea    0x68c8(%rip),%rax        # 100a0 <cbest_9>
    97d8:	48 89 05 a9 7e 00 00 	mov    %rax,0x7ea9(%rip)        # 11688 <cbest_all+0x48>
    97df:	48 8d 05 ba 58 00 00 	lea    0x58ba(%rip),%rax        # f0a0 <cbest_10>
    97e6:	48 89 05 a3 7e 00 00 	mov    %rax,0x7ea3(%rip)        # 11690 <cbest_all+0x50>
    97ed:	48 8d 05 ac 48 00 00 	lea    0x48ac(%rip),%rax        # e0a0 <cbest_11>
    97f4:	48 c7 05 41 7e 00 00 	movq   $0x0,0x7e41(%rip)        # 11640 <cbest_all>
    97fb:	00 00 00 00 
    97ff:	48 c7 05 3e 7e 00 00 	movq   $0x0,0x7e3e(%rip)        # 11648 <cbest_all+0x8>
    9806:	00 00 00 00 
    980a:	48 89 05 87 7e 00 00 	mov    %rax,0x7e87(%rip)        # 11698 <cbest_all+0x58>
    9811:	48 c7 05 84 7e 00 00 	movq   $0x0,0x7e84(%rip)        # 116a0 <cbest_all+0x60>
    9818:	00 00 00 00 
    981c:	48 c7 05 81 7e 00 00 	movq   $0x0,0x7e81(%rip)        # 116a8 <cbest_all+0x68>
    9823:	00 00 00 00 
    9827:	48 c7 05 7e 7e 00 00 	movq   $0x0,0x7e7e(%rip)        # 116b0 <cbest_all+0x70>
    982e:	00 00 00 00 
    9832:	48 c7 05 7b 7e 00 00 	movq   $0x0,0x7e7b(%rip)        # 116b8 <cbest_all+0x78>
    9839:	00 00 00 00 
    983d:	48 c7 05 78 7e 00 00 	movq   $0x0,0x7e78(%rip)        # 116c0 <cbest_all+0x80>
    9844:	00 00 00 00 
    9848:	48 c7 05 75 7e 00 00 	movq   $0x0,0x7e75(%rip)        # 116c8 <cbest_all+0x88>
    984f:	00 00 00 00 
    9853:	48 c7 05 72 7e 00 00 	movq   $0x0,0x7e72(%rip)        # 116d0 <cbest_all+0x90>
    985a:	00 00 00 00 
    985e:	48 c7 05 6f 7e 00 00 	movq   $0x0,0x7e6f(%rip)        # 116d8 <cbest_all+0x98>
    9865:	00 00 00 00 
    9869:	48 c7 05 6c 7e 00 00 	movq   $0x0,0x7e6c(%rip)        # 116e0 <cbest_all+0xa0>
    9870:	00 00 00 00 
    9874:	48 c7 05 69 7e 00 00 	movq   $0x0,0x7e69(%rip)        # 116e8 <cbest_all+0xa8>
    987b:	00 00 00 00 
    987f:	48 c7 05 66 7e 00 00 	movq   $0x0,0x7e66(%rip)        # 116f0 <cbest_all+0xb0>
    9886:	00 00 00 00 
    988a:	48 c7 05 63 7e 00 00 	movq   $0x0,0x7e63(%rip)        # 116f8 <cbest_all+0xb8>
    9891:	00 00 00 00 
    9895:	48 c7 05 60 7e 00 00 	movq   $0x0,0x7e60(%rip)        # 11700 <cbest_all+0xc0>
    989c:	00 00 00 00 
    98a0:	48 c7 05 5d 7e 00 00 	movq   $0x0,0x7e5d(%rip)        # 11708 <cbest_all+0xc8>
    98a7:	00 00 00 00 
    98ab:	48 c7 05 5a 7e 00 00 	movq   $0x0,0x7e5a(%rip)        # 11710 <cbest_all+0xd0>
    98b2:	00 00 00 00 
    98b6:	48 c7 05 57 7e 00 00 	movq   $0x0,0x7e57(%rip)        # 11718 <cbest_all+0xd8>
    98bd:	00 00 00 00 
    98c1:	48 c7 05 54 7e 00 00 	movq   $0x0,0x7e54(%rip)        # 11720 <cbest_all+0xe0>
    98c8:	00 00 00 00 
    98cc:	48 c7 05 51 7e 00 00 	movq   $0x0,0x7e51(%rip)        # 11728 <cbest_all+0xe8>
    98d3:	00 00 00 00 
    98d7:	48 c7 05 4e 7e 00 00 	movq   $0x0,0x7e4e(%rip)        # 11730 <cbest_all+0xf0>
    98de:	00 00 00 00 
    98e2:	48 c7 05 4b 7e 00 00 	movq   $0x0,0x7e4b(%rip)        # 11738 <cbest_all+0xf8>
    98e9:	00 00 00 00 
    98ed:	48 c7 05 48 7e 00 00 	movq   $0x0,0x7e48(%rip)        # 11740 <cbest_all+0x100>
    98f4:	00 00 00 00 
    98f8:	45 85 f6             	test   %r14d,%r14d
    98fb:	0f 8e 2c fe ff ff    	jle    972d <cauchy_good_general_coding_matrix+0x4d>
    9901:	48 8d 05 38 7d 00 00 	lea    0x7d38(%rip),%rax        # 11640 <cbest_all>
    9908:	49 63 fe             	movslq %r14d,%rdi
    990b:	41 8d 76 ff          	lea    -0x1(%r14),%esi
    990f:	31 d2                	xor    %edx,%edx
    9911:	4c 8b 04 d8          	mov    (%rax,%rbx,8),%r8
    9915:	49 8d 0c bc          	lea    (%r12,%rdi,4),%rcx
    9919:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    9920:	41 c7 04 94 01 00 00 	movl   $0x1,(%r12,%rdx,4)
    9927:	00 
    9928:	41 8b 04 90          	mov    (%r8,%rdx,4),%eax
    992c:	89 04 91             	mov    %eax,(%rcx,%rdx,4)
    992f:	48 89 d0             	mov    %rdx,%rax
    9932:	48 83 c2 01          	add    $0x1,%rdx
    9936:	48 39 c6             	cmp    %rax,%rsi
    9939:	75 e5                	jne    9920 <cauchy_good_general_coding_matrix+0x240>
    993b:	5b                   	pop    %rbx
    993c:	4c 89 e0             	mov    %r12,%rax
    993f:	5d                   	pop    %rbp
    9940:	41 5c                	pop    %r12
    9942:	41 5d                	pop    %r13
    9944:	41 5e                	pop    %r14
    9946:	c3                   	retq   
    9947:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    994e:	00 00 

0000000000009950 <timing_delta>:
    9950:	f3 0f 1e fa          	endbr64 
    9954:	41 57                	push   %r15
    9956:	49 89 f7             	mov    %rsi,%r15
    9959:	41 56                	push   %r14
    995b:	41 55                	push   %r13
    995d:	41 54                	push   %r12
    995f:	55                   	push   %rbp
    9960:	53                   	push   %rbx
    9961:	48 83 ec 68          	sub    $0x68,%rsp
    9965:	f2 0f 10 05 9b 8f 00 	movsd  0x8f9b(%rip),%xmm0        # 12908 <timing_time.2108>
    996c:	00 
    996d:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
    9972:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    9979:	00 00 
    997b:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    9980:	31 c0                	xor    %eax,%eax
    9982:	66 0f 2e 05 4e 0d 00 	ucomisd 0xd4e(%rip),%xmm0        # a6d8 <__PRETTY_FUNCTION__.5230+0x7>
    9989:	00 
    998a:	7a 74                	jp     9a00 <timing_delta+0xb0>
    998c:	75 72                	jne    9a00 <timing_delta+0xb0>
    998e:	bb 10 27 00 00       	mov    $0x2710,%ebx
    9993:	4c 8d 74 24 10       	lea    0x10(%rsp),%r14
    9998:	4c 8d 6c 24 30       	lea    0x30(%rsp),%r13
    999d:	4c 8d 64 24 40       	lea    0x40(%rsp),%r12
    99a2:	48 8d 6c 24 20       	lea    0x20(%rsp),%rbp
    99a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    99ae:	00 00 
    99b0:	4c 89 f6             	mov    %r14,%rsi
    99b3:	31 ff                	xor    %edi,%edi
    99b5:	e8 16 79 ff ff       	callq  12d0 <clock_gettime@plt>
    99ba:	4c 89 ee             	mov    %r13,%rsi
    99bd:	31 ff                	xor    %edi,%edi
    99bf:	e8 0c 79 ff ff       	callq  12d0 <clock_gettime@plt>
    99c4:	4c 89 e6             	mov    %r12,%rsi
    99c7:	31 ff                	xor    %edi,%edi
    99c9:	e8 02 79 ff ff       	callq  12d0 <clock_gettime@plt>
    99ce:	48 89 ee             	mov    %rbp,%rsi
    99d1:	31 ff                	xor    %edi,%edi
    99d3:	e8 f8 78 ff ff       	callq  12d0 <clock_gettime@plt>
    99d8:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    99dd:	66 0f ef c0          	pxor   %xmm0,%xmm0
    99e1:	48 2b 44 24 18       	sub    0x18(%rsp),%rax
    99e6:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    99eb:	f2 0f 5e 05 55 14 00 	divsd  0x1455(%rip),%xmm0        # ae48 <cbest_max_k+0x88>
    99f2:	00 
    99f3:	f2 0f 11 05 0d 8f 00 	movsd  %xmm0,0x8f0d(%rip)        # 12908 <timing_time.2108>
    99fa:	00 
    99fb:	83 eb 01             	sub    $0x1,%ebx
    99fe:	75 b0                	jne    99b0 <timing_delta+0x60>
    9a00:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
    9a05:	49 8b 47 08          	mov    0x8(%r15),%rax
    9a09:	66 0f ef c9          	pxor   %xmm1,%xmm1
    9a0d:	66 0f ef d2          	pxor   %xmm2,%xmm2
    9a11:	48 2b 42 08          	sub    0x8(%rdx),%rax
    9a15:	f2 48 0f 2a 12       	cvtsi2sdq (%rdx),%xmm2
    9a1a:	f2 48 0f 2a c8       	cvtsi2sd %rax,%xmm1
    9a1f:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
    9a24:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    9a2b:	00 00 
    9a2d:	f2 0f 5c c8          	subsd  %xmm0,%xmm1
    9a31:	66 0f 28 c1          	movapd %xmm1,%xmm0
    9a35:	66 0f ef c9          	pxor   %xmm1,%xmm1
    9a39:	f2 0f 5e 05 0f 14 00 	divsd  0x140f(%rip),%xmm0        # ae50 <cbest_max_k+0x90>
    9a40:	00 
    9a41:	f2 49 0f 2a 0f       	cvtsi2sdq (%r15),%xmm1
    9a46:	f2 0f 5c ca          	subsd  %xmm2,%xmm1
    9a4a:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
    9a4e:	75 0f                	jne    9a5f <timing_delta+0x10f>
    9a50:	48 83 c4 68          	add    $0x68,%rsp
    9a54:	5b                   	pop    %rbx
    9a55:	5d                   	pop    %rbp
    9a56:	41 5c                	pop    %r12
    9a58:	41 5d                	pop    %r13
    9a5a:	41 5e                	pop    %r14
    9a5c:	41 5f                	pop    %r15
    9a5e:	c3                   	retq   
    9a5f:	e8 9c 78 ff ff       	callq  1300 <__stack_chk_fail@plt>
    9a64:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    9a6b:	00 00 00 
    9a6e:	66 90                	xchg   %ax,%ax

0000000000009a70 <__libc_csu_init>:
    9a70:	f3 0f 1e fa          	endbr64 
    9a74:	41 57                	push   %r15
    9a76:	4c 8d 3d 2b 42 00 00 	lea    0x422b(%rip),%r15        # dca8 <__frame_dummy_init_array_entry>
    9a7d:	41 56                	push   %r14
    9a7f:	49 89 d6             	mov    %rdx,%r14
    9a82:	41 55                	push   %r13
    9a84:	49 89 f5             	mov    %rsi,%r13
    9a87:	41 54                	push   %r12
    9a89:	41 89 fc             	mov    %edi,%r12d
    9a8c:	55                   	push   %rbp
    9a8d:	48 8d 2d 1c 42 00 00 	lea    0x421c(%rip),%rbp        # dcb0 <__do_global_dtors_aux_fini_array_entry>
    9a94:	53                   	push   %rbx
    9a95:	4c 29 fd             	sub    %r15,%rbp
    9a98:	48 83 ec 08          	sub    $0x8,%rsp
    9a9c:	e8 5f 75 ff ff       	callq  1000 <_init>
    9aa1:	48 c1 fd 03          	sar    $0x3,%rbp
    9aa5:	74 1f                	je     9ac6 <__libc_csu_init+0x56>
    9aa7:	31 db                	xor    %ebx,%ebx
    9aa9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    9ab0:	4c 89 f2             	mov    %r14,%rdx
    9ab3:	4c 89 ee             	mov    %r13,%rsi
    9ab6:	44 89 e7             	mov    %r12d,%edi
    9ab9:	41 ff 14 df          	callq  *(%r15,%rbx,8)
    9abd:	48 83 c3 01          	add    $0x1,%rbx
    9ac1:	48 39 dd             	cmp    %rbx,%rbp
    9ac4:	75 ea                	jne    9ab0 <__libc_csu_init+0x40>
    9ac6:	48 83 c4 08          	add    $0x8,%rsp
    9aca:	5b                   	pop    %rbx
    9acb:	5d                   	pop    %rbp
    9acc:	41 5c                	pop    %r12
    9ace:	41 5d                	pop    %r13
    9ad0:	41 5e                	pop    %r14
    9ad2:	41 5f                	pop    %r15
    9ad4:	c3                   	retq   
    9ad5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    9adc:	00 00 00 00 

0000000000009ae0 <__libc_csu_fini>:
    9ae0:	f3 0f 1e fa          	endbr64 
    9ae4:	c3                   	retq   

Disassembly of section .fini:

0000000000009ae8 <_fini>:
    9ae8:	f3 0f 1e fa          	endbr64 
    9aec:	48 83 ec 08          	sub    $0x8,%rsp
    9af0:	48 83 c4 08          	add    $0x8,%rsp
    9af4:	c3                   	retq   

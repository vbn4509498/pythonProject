��	
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-15-g6290819256d8ͷ
h
Adam/b/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
Adam/b/v
a
Adam/b/v/Read/ReadVariableOpReadVariableOpAdam/b/v*
_output_shapes
:
*
dtype0
l
Adam/w/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
*
shared_name
Adam/w/v
e
Adam/w/v/Read/ReadVariableOpReadVariableOpAdam/w/v*
_output_shapes

: 
*
dtype0
l

Adam/b/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/b/v_1
e
Adam/b/v_1/Read/ReadVariableOpReadVariableOp
Adam/b/v_1*
_output_shapes
: *
dtype0
p

Adam/w/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_name
Adam/w/v_1
i
Adam/w/v_1/Read/ReadVariableOpReadVariableOp
Adam/w/v_1*
_output_shapes

:@ *
dtype0
l

Adam/b/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
Adam/b/v_2
e
Adam/b/v_2/Read/ReadVariableOpReadVariableOp
Adam/b/v_2*
_output_shapes
:@*
dtype0
q

Adam/w/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_name
Adam/w/v_2
j
Adam/w/v_2/Read/ReadVariableOpReadVariableOp
Adam/w/v_2*
_output_shapes
:	�@*
dtype0
m

Adam/b/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
Adam/b/v_3
f
Adam/b/v_3/Read/ReadVariableOpReadVariableOp
Adam/b/v_3*
_output_shapes	
:�*
dtype0
r

Adam/w/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name
Adam/w/v_3
k
Adam/w/v_3/Read/ReadVariableOpReadVariableOp
Adam/w/v_3* 
_output_shapes
:
��*
dtype0
m

Adam/b/v_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
Adam/b/v_4
f
Adam/b/v_4/Read/ReadVariableOpReadVariableOp
Adam/b/v_4*
_output_shapes	
:�*
dtype0
r

Adam/w/v_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name
Adam/w/v_4
k
Adam/w/v_4/Read/ReadVariableOpReadVariableOp
Adam/w/v_4* 
_output_shapes
:
��*
dtype0
h
Adam/b/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
Adam/b/m
a
Adam/b/m/Read/ReadVariableOpReadVariableOpAdam/b/m*
_output_shapes
:
*
dtype0
l
Adam/w/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
*
shared_name
Adam/w/m
e
Adam/w/m/Read/ReadVariableOpReadVariableOpAdam/w/m*
_output_shapes

: 
*
dtype0
l

Adam/b/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/b/m_1
e
Adam/b/m_1/Read/ReadVariableOpReadVariableOp
Adam/b/m_1*
_output_shapes
: *
dtype0
p

Adam/w/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_name
Adam/w/m_1
i
Adam/w/m_1/Read/ReadVariableOpReadVariableOp
Adam/w/m_1*
_output_shapes

:@ *
dtype0
l

Adam/b/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
Adam/b/m_2
e
Adam/b/m_2/Read/ReadVariableOpReadVariableOp
Adam/b/m_2*
_output_shapes
:@*
dtype0
q

Adam/w/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_name
Adam/w/m_2
j
Adam/w/m_2/Read/ReadVariableOpReadVariableOp
Adam/w/m_2*
_output_shapes
:	�@*
dtype0
m

Adam/b/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
Adam/b/m_3
f
Adam/b/m_3/Read/ReadVariableOpReadVariableOp
Adam/b/m_3*
_output_shapes	
:�*
dtype0
r

Adam/w/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name
Adam/w/m_3
k
Adam/w/m_3/Read/ReadVariableOpReadVariableOp
Adam/w/m_3* 
_output_shapes
:
��*
dtype0
m

Adam/b/m_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
Adam/b/m_4
f
Adam/b/m_4/Read/ReadVariableOpReadVariableOp
Adam/b/m_4*
_output_shapes	
:�*
dtype0
r

Adam/w/m_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name
Adam/w/m_4
k
Adam/w/m_4/Read/ReadVariableOpReadVariableOp
Adam/w/m_4* 
_output_shapes
:
��*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
Z
bVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameb
S
b/Read/ReadVariableOpReadVariableOpb*
_output_shapes
:
*
dtype0
^
wVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
*
shared_namew
W
w/Read/ReadVariableOpReadVariableOpw*
_output_shapes

: 
*
dtype0
^
b_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameb_1
W
b_1/Read/ReadVariableOpReadVariableOpb_1*
_output_shapes
: *
dtype0
b
w_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namew_1
[
w_1/Read/ReadVariableOpReadVariableOpw_1*
_output_shapes

:@ *
dtype0
^
b_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameb_2
W
b_2/Read/ReadVariableOpReadVariableOpb_2*
_output_shapes
:@*
dtype0
c
w_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namew_2
\
w_2/Read/ReadVariableOpReadVariableOpw_2*
_output_shapes
:	�@*
dtype0
_
b_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameb_3
X
b_3/Read/ReadVariableOpReadVariableOpb_3*
_output_shapes	
:�*
dtype0
d
w_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namew_3
]
w_3/Read/ReadVariableOpReadVariableOpw_3* 
_output_shapes
:
��*
dtype0
_
b_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameb_4
X
b_4/Read/ReadVariableOpReadVariableOpb_4*
_output_shapes	
:�*
dtype0
d
w_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namew_4
]
w_4/Read/ReadVariableOpReadVariableOpw_4* 
_output_shapes
:
��*
dtype0
|
serving_default_input_1Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1w_4b_4w_3b_3w_2b_2w_1b_1wb*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_12238

NoOpNoOp
�?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�>
value�>B�> B�>
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

fc3
fc4
fc5
	optimizer

signatures*
J
0
1
2
3
4
5
6
7
8
9*
J
0
1
2
3
4
5
6
7
8
9*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
 trace_2
!trace_3* 
6
"trace_0
#trace_1
$trace_2
%trace_3* 
* 
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
w

kernal
b
bias*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
w

kernal
b
bias*
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
w

kernal
b
bias*
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
w

kernal
b
bias*
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
w

kernal
b
bias*
�
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_ratem�m�m�m�m�m�m�m�m�m�v�v�v�v�v�v�v�v�v�v�*

Iserving_default* 
C=
VARIABLE_VALUEw_4&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
C=
VARIABLE_VALUEb_4&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
C=
VARIABLE_VALUEw_3&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
C=
VARIABLE_VALUEb_3&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
C=
VARIABLE_VALUEw_2&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
C=
VARIABLE_VALUEb_2&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
C=
VARIABLE_VALUEw_1&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
C=
VARIABLE_VALUEb_1&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
A;
VARIABLE_VALUEw&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
A;
VARIABLE_VALUEb&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
	1

2
3
4*

J0
K1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

Qtrace_0
Rtrace_1* 

Strace_0
Ttrace_1* 

0
1*

0
1*
* 
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

Ztrace_0
[trace_1* 

\trace_0
]trace_1* 

0
1*

0
1*
* 
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

ctrace_0
dtrace_1* 

etrace_0
ftrace_1* 

0
1*

0
1*
* 
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

ltrace_0
mtrace_1* 

ntrace_0
otrace_1* 

0
1*

0
1*
* 
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

utrace_0
vtrace_1* 

wtrace_0
xtrace_1* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
8
y	variables
z	keras_api
	{total
	|count*
J
}	variables
~	keras_api
	total

�count
�
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

{0
|1*

y	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
�1*

}	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
f`
VARIABLE_VALUE
Adam/w/m_4Bvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/b/m_4Bvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/w/m_3Bvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/b/m_3Bvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/w/m_2Bvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/b/m_2Bvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/w/m_1Bvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/b/m_1Bvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/w/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/b/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/w/v_4Bvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/b/v_4Bvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/w/v_3Bvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/b/v_3Bvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/w/v_2Bvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/b/v_2Bvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/w/v_1Bvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE
Adam/b/v_1Bvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/w/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/b/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamew_4/Read/ReadVariableOpb_4/Read/ReadVariableOpw_3/Read/ReadVariableOpb_3/Read/ReadVariableOpw_2/Read/ReadVariableOpb_2/Read/ReadVariableOpw_1/Read/ReadVariableOpb_1/Read/ReadVariableOpw/Read/ReadVariableOpb/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpAdam/w/m_4/Read/ReadVariableOpAdam/b/m_4/Read/ReadVariableOpAdam/w/m_3/Read/ReadVariableOpAdam/b/m_3/Read/ReadVariableOpAdam/w/m_2/Read/ReadVariableOpAdam/b/m_2/Read/ReadVariableOpAdam/w/m_1/Read/ReadVariableOpAdam/b/m_1/Read/ReadVariableOpAdam/w/m/Read/ReadVariableOpAdam/b/m/Read/ReadVariableOpAdam/w/v_4/Read/ReadVariableOpAdam/b/v_4/Read/ReadVariableOpAdam/w/v_3/Read/ReadVariableOpAdam/b/v_3/Read/ReadVariableOpAdam/w/v_2/Read/ReadVariableOpAdam/b/v_2/Read/ReadVariableOpAdam/w/v_1/Read/ReadVariableOpAdam/b/v_1/Read/ReadVariableOpAdam/w/v/Read/ReadVariableOpAdam/b/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_12694
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamew_4b_4w_3b_3w_2b_2w_1b_1wb	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount
Adam/w/m_4
Adam/b/m_4
Adam/w/m_3
Adam/b/m_3
Adam/w/m_2
Adam/b/m_2
Adam/w/m_1
Adam/b/m_1Adam/w/mAdam/b/m
Adam/w/v_4
Adam/b/v_4
Adam/w/v_3
Adam/b/v_3
Adam/w/v_2
Adam/b/v_2
Adam/w/v_1
Adam/b/v_1Adam/w/vAdam/b/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_12821��
�	
�
C__inference_my_dense_layer_call_and_return_conditional_losses_12392

inputs2
matmul_readvariableop_resource:
��*
add_readvariableop_resource:	�
identity��add/ReadVariableOp�matmul/ReadVariableOpv
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0m
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
IdentityIdentityadd:z:0^NoOp*
T0*(
_output_shapes
:����������s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_2_layer_call_and_return_conditional_losses_11822

inputs1
matmul_readvariableop_resource:	�@)
add_readvariableop_resource:@
identity��add/ReadVariableOp�matmul/ReadVariableOpu
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0l
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_my_dense_layer_call_fn_12373

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_my_dense_layer_call_and_return_conditional_losses_11788p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_2_layer_call_and_return_conditional_losses_12478

inputs1
matmul_readvariableop_resource:	�@)
add_readvariableop_resource:@
identity��add/ReadVariableOp�matmul/ReadVariableOpu
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0l
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
(__inference_my_model_layer_call_fn_12288

inputs
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 

	unknown_8:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_my_model_layer_call_and_return_conditional_losses_12091o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_1_layer_call_and_return_conditional_losses_12440

inputs2
matmul_readvariableop_resource:
��*
add_readvariableop_resource:	�
identity��add/ReadVariableOp�matmul/ReadVariableOpv
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0m
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
IdentityIdentityadd:z:0^NoOp*
T0*(
_output_shapes
:����������s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_1_layer_call_and_return_conditional_losses_11805

inputs2
matmul_readvariableop_resource:
��*
add_readvariableop_resource:	�
identity��add/ReadVariableOp�matmul/ReadVariableOpv
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0m
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
IdentityIdentityadd:z:0^NoOp*
T0*(
_output_shapes
:����������s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_my_model_layer_call_and_return_conditional_losses_11863

inputs"
my_dense_11789:
��
my_dense_11791:	�$
my_dense_1_11806:
��
my_dense_1_11808:	�#
my_dense_2_11823:	�@
my_dense_2_11825:@"
my_dense_3_11840:@ 
my_dense_3_11842: "
my_dense_4_11857: 

my_dense_4_11859:

identity�� my_dense/StatefulPartitionedCall�"my_dense_1/StatefulPartitionedCall�"my_dense_2/StatefulPartitionedCall�"my_dense_3/StatefulPartitionedCall�"my_dense_4/StatefulPartitionedCall�
 my_dense/StatefulPartitionedCallStatefulPartitionedCallinputsmy_dense_11789my_dense_11791*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_my_dense_layer_call_and_return_conditional_losses_11788j
ReluRelu)my_dense/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:�����������
"my_dense_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0my_dense_1_11806my_dense_1_11808*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_1_layer_call_and_return_conditional_losses_11805n
Relu_1Relu+my_dense_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:�����������
"my_dense_2/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0my_dense_2_11823my_dense_2_11825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_2_layer_call_and_return_conditional_losses_11822m
Relu_2Relu+my_dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@�
"my_dense_3/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0my_dense_3_11840my_dense_3_11842*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_3_layer_call_and_return_conditional_losses_11839m
Relu_3Relu+my_dense_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:��������� �
"my_dense_4/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0my_dense_4_11857my_dense_4_11859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_4_layer_call_and_return_conditional_losses_11856z
IdentityIdentity+my_dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp!^my_dense/StatefulPartitionedCall#^my_dense_1/StatefulPartitionedCall#^my_dense_2/StatefulPartitionedCall#^my_dense_3/StatefulPartitionedCall#^my_dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2D
 my_dense/StatefulPartitionedCall my_dense/StatefulPartitionedCall2H
"my_dense_1/StatefulPartitionedCall"my_dense_1/StatefulPartitionedCall2H
"my_dense_2/StatefulPartitionedCall"my_dense_2/StatefulPartitionedCall2H
"my_dense_3/StatefulPartitionedCall"my_dense_3/StatefulPartitionedCall2H
"my_dense_4/StatefulPartitionedCall"my_dense_4/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
(__inference_my_model_layer_call_fn_12139
input_1
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 

	unknown_8:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_my_model_layer_call_and_return_conditional_losses_12091o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�	
�
E__inference_my_dense_4_layer_call_and_return_conditional_losses_12554

inputs0
matmul_readvariableop_resource: 
)
add_readvariableop_resource:

identity��add/ReadVariableOp�matmul/ReadVariableOpt
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype0i
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:
*
dtype0l
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������
s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
C__inference_my_model_layer_call_and_return_conditional_losses_12172
input_1"
my_dense_12142:
��
my_dense_12144:	�$
my_dense_1_12148:
��
my_dense_1_12150:	�#
my_dense_2_12154:	�@
my_dense_2_12156:@"
my_dense_3_12160:@ 
my_dense_3_12162: "
my_dense_4_12166: 

my_dense_4_12168:

identity�� my_dense/StatefulPartitionedCall�"my_dense_1/StatefulPartitionedCall�"my_dense_2/StatefulPartitionedCall�"my_dense_3/StatefulPartitionedCall�"my_dense_4/StatefulPartitionedCall�
 my_dense/StatefulPartitionedCallStatefulPartitionedCallinput_1my_dense_12142my_dense_12144*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_my_dense_layer_call_and_return_conditional_losses_11788j
ReluRelu)my_dense/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:�����������
"my_dense_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0my_dense_1_12148my_dense_1_12150*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_1_layer_call_and_return_conditional_losses_11805n
Relu_1Relu+my_dense_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:�����������
"my_dense_2/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0my_dense_2_12154my_dense_2_12156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_2_layer_call_and_return_conditional_losses_11822m
Relu_2Relu+my_dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@�
"my_dense_3/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0my_dense_3_12160my_dense_3_12162*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_3_layer_call_and_return_conditional_losses_11839m
Relu_3Relu+my_dense_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:��������� �
"my_dense_4/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0my_dense_4_12166my_dense_4_12168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_4_layer_call_and_return_conditional_losses_11856z
IdentityIdentity+my_dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp!^my_dense/StatefulPartitionedCall#^my_dense_1/StatefulPartitionedCall#^my_dense_2/StatefulPartitionedCall#^my_dense_3/StatefulPartitionedCall#^my_dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2D
 my_dense/StatefulPartitionedCall my_dense/StatefulPartitionedCall2H
"my_dense_1/StatefulPartitionedCall"my_dense_1/StatefulPartitionedCall2H
"my_dense_2/StatefulPartitionedCall"my_dense_2/StatefulPartitionedCall2H
"my_dense_3/StatefulPartitionedCall"my_dense_3/StatefulPartitionedCall2H
"my_dense_4/StatefulPartitionedCall"my_dense_4/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�H
�
__inference__traced_save_12694
file_prefix"
savev2_w_4_read_readvariableop"
savev2_b_4_read_readvariableop"
savev2_w_3_read_readvariableop"
savev2_b_3_read_readvariableop"
savev2_w_2_read_readvariableop"
savev2_b_2_read_readvariableop"
savev2_w_1_read_readvariableop"
savev2_b_1_read_readvariableop 
savev2_w_read_readvariableop 
savev2_b_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop)
%savev2_adam_w_m_4_read_readvariableop)
%savev2_adam_b_m_4_read_readvariableop)
%savev2_adam_w_m_3_read_readvariableop)
%savev2_adam_b_m_3_read_readvariableop)
%savev2_adam_w_m_2_read_readvariableop)
%savev2_adam_b_m_2_read_readvariableop)
%savev2_adam_w_m_1_read_readvariableop)
%savev2_adam_b_m_1_read_readvariableop'
#savev2_adam_w_m_read_readvariableop'
#savev2_adam_b_m_read_readvariableop)
%savev2_adam_w_v_4_read_readvariableop)
%savev2_adam_b_v_4_read_readvariableop)
%savev2_adam_w_v_3_read_readvariableop)
%savev2_adam_b_v_3_read_readvariableop)
%savev2_adam_w_v_2_read_readvariableop)
%savev2_adam_b_v_2_read_readvariableop)
%savev2_adam_w_v_1_read_readvariableop)
%savev2_adam_b_v_1_read_readvariableop'
#savev2_adam_w_v_read_readvariableop'
#savev2_adam_b_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_w_4_read_readvariableopsavev2_b_4_read_readvariableopsavev2_w_3_read_readvariableopsavev2_b_3_read_readvariableopsavev2_w_2_read_readvariableopsavev2_b_2_read_readvariableopsavev2_w_1_read_readvariableopsavev2_b_1_read_readvariableopsavev2_w_read_readvariableopsavev2_b_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop%savev2_adam_w_m_4_read_readvariableop%savev2_adam_b_m_4_read_readvariableop%savev2_adam_w_m_3_read_readvariableop%savev2_adam_b_m_3_read_readvariableop%savev2_adam_w_m_2_read_readvariableop%savev2_adam_b_m_2_read_readvariableop%savev2_adam_w_m_1_read_readvariableop%savev2_adam_b_m_1_read_readvariableop#savev2_adam_w_m_read_readvariableop#savev2_adam_b_m_read_readvariableop%savev2_adam_w_v_4_read_readvariableop%savev2_adam_b_v_4_read_readvariableop%savev2_adam_w_v_3_read_readvariableop%savev2_adam_b_v_3_read_readvariableop%savev2_adam_w_v_2_read_readvariableop%savev2_adam_b_v_2_read_readvariableop%savev2_adam_w_v_1_read_readvariableop%savev2_adam_b_v_1_read_readvariableop#savev2_adam_w_v_read_readvariableop#savev2_adam_b_v_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *6
dtypes,
*2(	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:
��:�:	�@:@:@ : : 
:
: : : : : : : : : :
��:�:
��:�:	�@:@:@ : : 
:
:
��:�:
��:�:	�@:@:@ : : 
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$	 

_output_shapes

: 
: 


_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: 
: 

_output_shapes
:
:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:& "
 
_output_shapes
:
��:!!

_output_shapes	
:�:%"!

_output_shapes
:	�@: #

_output_shapes
:@:$$ 

_output_shapes

:@ : %

_output_shapes
: :$& 

_output_shapes

: 
: '

_output_shapes
:
:(

_output_shapes
: 
�	
�
E__inference_my_dense_4_layer_call_and_return_conditional_losses_12544

inputs0
matmul_readvariableop_resource: 
)
add_readvariableop_resource:

identity��add/ReadVariableOp�matmul/ReadVariableOpt
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype0i
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:
*
dtype0l
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������
s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_1_layer_call_and_return_conditional_losses_12430

inputs2
matmul_readvariableop_resource:
��*
add_readvariableop_resource:	�
identity��add/ReadVariableOp�matmul/ReadVariableOpv
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0m
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
IdentityIdentityadd:z:0^NoOp*
T0*(
_output_shapes
:����������s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
(__inference_my_model_layer_call_fn_11886
input_1
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 

	unknown_8:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_my_model_layer_call_and_return_conditional_losses_11863o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
*__inference_my_dense_3_layer_call_fn_12487

inputs
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_3_layer_call_and_return_conditional_losses_11839o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
C__inference_my_dense_layer_call_and_return_conditional_losses_11788

inputs2
matmul_readvariableop_resource:
��*
add_readvariableop_resource:	�
identity��add/ReadVariableOp�matmul/ReadVariableOpv
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0m
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
IdentityIdentityadd:z:0^NoOp*
T0*(
_output_shapes
:����������s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�+
�
C__inference_my_model_layer_call_and_return_conditional_losses_12326

inputs;
'my_dense_matmul_readvariableop_resource:
��3
$my_dense_add_readvariableop_resource:	�=
)my_dense_1_matmul_readvariableop_resource:
��5
&my_dense_1_add_readvariableop_resource:	�<
)my_dense_2_matmul_readvariableop_resource:	�@4
&my_dense_2_add_readvariableop_resource:@;
)my_dense_3_matmul_readvariableop_resource:@ 4
&my_dense_3_add_readvariableop_resource: ;
)my_dense_4_matmul_readvariableop_resource: 
4
&my_dense_4_add_readvariableop_resource:

identity��my_dense/add/ReadVariableOp�my_dense/matmul/ReadVariableOp�my_dense_1/add/ReadVariableOp� my_dense_1/matmul/ReadVariableOp�my_dense_2/add/ReadVariableOp� my_dense_2/matmul/ReadVariableOp�my_dense_3/add/ReadVariableOp� my_dense_3/matmul/ReadVariableOp�my_dense_4/add/ReadVariableOp� my_dense_4/matmul/ReadVariableOp�
my_dense/matmul/ReadVariableOpReadVariableOp'my_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0|
my_dense/matmulMatMulinputs&my_dense/matmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
my_dense/add/ReadVariableOpReadVariableOp$my_dense_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
my_dense/addAddV2my_dense/matmul:product:0#my_dense/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluRelumy_dense/add:z:0*
T0*(
_output_shapes
:�����������
 my_dense_1/matmul/ReadVariableOpReadVariableOp)my_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
my_dense_1/matmulMatMulRelu:activations:0(my_dense_1/matmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
my_dense_1/add/ReadVariableOpReadVariableOp&my_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
my_dense_1/addAddV2my_dense_1/matmul:product:0%my_dense_1/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������U
Relu_1Relumy_dense_1/add:z:0*
T0*(
_output_shapes
:�����������
 my_dense_2/matmul/ReadVariableOpReadVariableOp)my_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
my_dense_2/matmulMatMulRelu_1:activations:0(my_dense_2/matmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
my_dense_2/add/ReadVariableOpReadVariableOp&my_dense_2_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
my_dense_2/addAddV2my_dense_2/matmul:product:0%my_dense_2/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@T
Relu_2Relumy_dense_2/add:z:0*
T0*'
_output_shapes
:���������@�
 my_dense_3/matmul/ReadVariableOpReadVariableOp)my_dense_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
my_dense_3/matmulMatMulRelu_2:activations:0(my_dense_3/matmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
my_dense_3/add/ReadVariableOpReadVariableOp&my_dense_3_add_readvariableop_resource*
_output_shapes
: *
dtype0�
my_dense_3/addAddV2my_dense_3/matmul:product:0%my_dense_3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� T
Relu_3Relumy_dense_3/add:z:0*
T0*'
_output_shapes
:��������� �
 my_dense_4/matmul/ReadVariableOpReadVariableOp)my_dense_4_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype0�
my_dense_4/matmulMatMulRelu_3:activations:0(my_dense_4/matmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
my_dense_4/add/ReadVariableOpReadVariableOp&my_dense_4_add_readvariableop_resource*
_output_shapes
:
*
dtype0�
my_dense_4/addAddV2my_dense_4/matmul:product:0%my_dense_4/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
a
IdentityIdentitymy_dense_4/add:z:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^my_dense/add/ReadVariableOp^my_dense/matmul/ReadVariableOp^my_dense_1/add/ReadVariableOp!^my_dense_1/matmul/ReadVariableOp^my_dense_2/add/ReadVariableOp!^my_dense_2/matmul/ReadVariableOp^my_dense_3/add/ReadVariableOp!^my_dense_3/matmul/ReadVariableOp^my_dense_4/add/ReadVariableOp!^my_dense_4/matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2:
my_dense/add/ReadVariableOpmy_dense/add/ReadVariableOp2@
my_dense/matmul/ReadVariableOpmy_dense/matmul/ReadVariableOp2>
my_dense_1/add/ReadVariableOpmy_dense_1/add/ReadVariableOp2D
 my_dense_1/matmul/ReadVariableOp my_dense_1/matmul/ReadVariableOp2>
my_dense_2/add/ReadVariableOpmy_dense_2/add/ReadVariableOp2D
 my_dense_2/matmul/ReadVariableOp my_dense_2/matmul/ReadVariableOp2>
my_dense_3/add/ReadVariableOpmy_dense_3/add/ReadVariableOp2D
 my_dense_3/matmul/ReadVariableOp my_dense_3/matmul/ReadVariableOp2>
my_dense_4/add/ReadVariableOpmy_dense_4/add/ReadVariableOp2D
 my_dense_4/matmul/ReadVariableOp my_dense_4/matmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_my_dense_4_layer_call_fn_12534

inputs
unknown: 

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_4_layer_call_and_return_conditional_losses_11908o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_2_layer_call_and_return_conditional_losses_12468

inputs1
matmul_readvariableop_resource:	�@)
add_readvariableop_resource:@
identity��add/ReadVariableOp�matmul/ReadVariableOpu
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0l
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
!__inference__traced_restore_12821
file_prefix(
assignvariableop_w_4:
��%
assignvariableop_1_b_4:	�*
assignvariableop_2_w_3:
��%
assignvariableop_3_b_3:	�)
assignvariableop_4_w_2:	�@$
assignvariableop_5_b_2:@(
assignvariableop_6_w_1:@ $
assignvariableop_7_b_1: &
assignvariableop_8_w: 
"
assignvariableop_9_b:
'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: #
assignvariableop_17_total: #
assignvariableop_18_count: 2
assignvariableop_19_adam_w_m_4:
��-
assignvariableop_20_adam_b_m_4:	�2
assignvariableop_21_adam_w_m_3:
��-
assignvariableop_22_adam_b_m_3:	�1
assignvariableop_23_adam_w_m_2:	�@,
assignvariableop_24_adam_b_m_2:@0
assignvariableop_25_adam_w_m_1:@ ,
assignvariableop_26_adam_b_m_1: .
assignvariableop_27_adam_w_m: 
*
assignvariableop_28_adam_b_m:
2
assignvariableop_29_adam_w_v_4:
��-
assignvariableop_30_adam_b_v_4:	�2
assignvariableop_31_adam_w_v_3:
��-
assignvariableop_32_adam_b_v_3:	�1
assignvariableop_33_adam_w_v_2:	�@,
assignvariableop_34_adam_b_v_2:@0
assignvariableop_35_adam_w_v_1:@ ,
assignvariableop_36_adam_b_v_1: .
assignvariableop_37_adam_w_v: 
*
assignvariableop_38_adam_b_v:

identity_40��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_w_4Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_b_4Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_w_3Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_b_3Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_w_2Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_b_2Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_w_1Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_b_1Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_wIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_bIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_w_m_4Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_b_m_4Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_w_m_3Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_b_m_3Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_w_m_2Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_b_m_2Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_w_m_1Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_b_m_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_w_mIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_b_mIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_w_v_4Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_b_v_4Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_w_v_3Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_b_v_3Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_w_v_2Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_b_v_2Identity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_w_v_1Identity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_adam_b_v_1Identity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_adam_w_vIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_b_vIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
*__inference_my_dense_2_layer_call_fn_12449

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_2_layer_call_and_return_conditional_losses_11822o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�2
�	
 __inference__wrapped_model_11771
input_1D
0my_model_my_dense_matmul_readvariableop_resource:
��<
-my_model_my_dense_add_readvariableop_resource:	�F
2my_model_my_dense_1_matmul_readvariableop_resource:
��>
/my_model_my_dense_1_add_readvariableop_resource:	�E
2my_model_my_dense_2_matmul_readvariableop_resource:	�@=
/my_model_my_dense_2_add_readvariableop_resource:@D
2my_model_my_dense_3_matmul_readvariableop_resource:@ =
/my_model_my_dense_3_add_readvariableop_resource: D
2my_model_my_dense_4_matmul_readvariableop_resource: 
=
/my_model_my_dense_4_add_readvariableop_resource:

identity��$my_model/my_dense/add/ReadVariableOp�'my_model/my_dense/matmul/ReadVariableOp�&my_model/my_dense_1/add/ReadVariableOp�)my_model/my_dense_1/matmul/ReadVariableOp�&my_model/my_dense_2/add/ReadVariableOp�)my_model/my_dense_2/matmul/ReadVariableOp�&my_model/my_dense_3/add/ReadVariableOp�)my_model/my_dense_3/matmul/ReadVariableOp�&my_model/my_dense_4/add/ReadVariableOp�)my_model/my_dense_4/matmul/ReadVariableOp�
'my_model/my_dense/matmul/ReadVariableOpReadVariableOp0my_model_my_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
my_model/my_dense/matmulMatMulinput_1/my_model/my_dense/matmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$my_model/my_dense/add/ReadVariableOpReadVariableOp-my_model_my_dense_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
my_model/my_dense/addAddV2"my_model/my_dense/matmul:product:0,my_model/my_dense/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
my_model/ReluRelumy_model/my_dense/add:z:0*
T0*(
_output_shapes
:�����������
)my_model/my_dense_1/matmul/ReadVariableOpReadVariableOp2my_model_my_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
my_model/my_dense_1/matmulMatMulmy_model/Relu:activations:01my_model/my_dense_1/matmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&my_model/my_dense_1/add/ReadVariableOpReadVariableOp/my_model_my_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
my_model/my_dense_1/addAddV2$my_model/my_dense_1/matmul:product:0.my_model/my_dense_1/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
my_model/Relu_1Relumy_model/my_dense_1/add:z:0*
T0*(
_output_shapes
:�����������
)my_model/my_dense_2/matmul/ReadVariableOpReadVariableOp2my_model_my_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
my_model/my_dense_2/matmulMatMulmy_model/Relu_1:activations:01my_model/my_dense_2/matmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&my_model/my_dense_2/add/ReadVariableOpReadVariableOp/my_model_my_dense_2_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
my_model/my_dense_2/addAddV2$my_model/my_dense_2/matmul:product:0.my_model/my_dense_2/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@f
my_model/Relu_2Relumy_model/my_dense_2/add:z:0*
T0*'
_output_shapes
:���������@�
)my_model/my_dense_3/matmul/ReadVariableOpReadVariableOp2my_model_my_dense_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
my_model/my_dense_3/matmulMatMulmy_model/Relu_2:activations:01my_model/my_dense_3/matmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&my_model/my_dense_3/add/ReadVariableOpReadVariableOp/my_model_my_dense_3_add_readvariableop_resource*
_output_shapes
: *
dtype0�
my_model/my_dense_3/addAddV2$my_model/my_dense_3/matmul:product:0.my_model/my_dense_3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� f
my_model/Relu_3Relumy_model/my_dense_3/add:z:0*
T0*'
_output_shapes
:��������� �
)my_model/my_dense_4/matmul/ReadVariableOpReadVariableOp2my_model_my_dense_4_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype0�
my_model/my_dense_4/matmulMatMulmy_model/Relu_3:activations:01my_model/my_dense_4/matmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
&my_model/my_dense_4/add/ReadVariableOpReadVariableOp/my_model_my_dense_4_add_readvariableop_resource*
_output_shapes
:
*
dtype0�
my_model/my_dense_4/addAddV2$my_model/my_dense_4/matmul:product:0.my_model/my_dense_4/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
j
IdentityIdentitymy_model/my_dense_4/add:z:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp%^my_model/my_dense/add/ReadVariableOp(^my_model/my_dense/matmul/ReadVariableOp'^my_model/my_dense_1/add/ReadVariableOp*^my_model/my_dense_1/matmul/ReadVariableOp'^my_model/my_dense_2/add/ReadVariableOp*^my_model/my_dense_2/matmul/ReadVariableOp'^my_model/my_dense_3/add/ReadVariableOp*^my_model/my_dense_3/matmul/ReadVariableOp'^my_model/my_dense_4/add/ReadVariableOp*^my_model/my_dense_4/matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2L
$my_model/my_dense/add/ReadVariableOp$my_model/my_dense/add/ReadVariableOp2R
'my_model/my_dense/matmul/ReadVariableOp'my_model/my_dense/matmul/ReadVariableOp2P
&my_model/my_dense_1/add/ReadVariableOp&my_model/my_dense_1/add/ReadVariableOp2V
)my_model/my_dense_1/matmul/ReadVariableOp)my_model/my_dense_1/matmul/ReadVariableOp2P
&my_model/my_dense_2/add/ReadVariableOp&my_model/my_dense_2/add/ReadVariableOp2V
)my_model/my_dense_2/matmul/ReadVariableOp)my_model/my_dense_2/matmul/ReadVariableOp2P
&my_model/my_dense_3/add/ReadVariableOp&my_model/my_dense_3/add/ReadVariableOp2V
)my_model/my_dense_3/matmul/ReadVariableOp)my_model/my_dense_3/matmul/ReadVariableOp2P
&my_model/my_dense_4/add/ReadVariableOp&my_model/my_dense_4/add/ReadVariableOp2V
)my_model/my_dense_4/matmul/ReadVariableOp)my_model/my_dense_4/matmul/ReadVariableOp:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�	
�
E__inference_my_dense_4_layer_call_and_return_conditional_losses_11856

inputs0
matmul_readvariableop_resource: 
)
add_readvariableop_resource:

identity��add/ReadVariableOp�matmul/ReadVariableOpt
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype0i
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:
*
dtype0l
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������
s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
*__inference_my_dense_4_layer_call_fn_12525

inputs
unknown: 

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_4_layer_call_and_return_conditional_losses_11856o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_3_layer_call_and_return_conditional_losses_12516

inputs0
matmul_readvariableop_resource:@ )
add_readvariableop_resource: 
identity��add/ReadVariableOp�matmul/ReadVariableOpt
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0l
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:��������� s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
#__inference_signature_wrapper_12238
input_1
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 

	unknown_8:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_11771o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�

�
(__inference_my_model_layer_call_fn_12263

inputs
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 

	unknown_8:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_my_model_layer_call_and_return_conditional_losses_11863o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_my_model_layer_call_and_return_conditional_losses_12205
input_1"
my_dense_12175:
��
my_dense_12177:	�$
my_dense_1_12181:
��
my_dense_1_12183:	�#
my_dense_2_12187:	�@
my_dense_2_12189:@"
my_dense_3_12193:@ 
my_dense_3_12195: "
my_dense_4_12199: 

my_dense_4_12201:

identity�� my_dense/StatefulPartitionedCall�"my_dense_1/StatefulPartitionedCall�"my_dense_2/StatefulPartitionedCall�"my_dense_3/StatefulPartitionedCall�"my_dense_4/StatefulPartitionedCall�
 my_dense/StatefulPartitionedCallStatefulPartitionedCallinput_1my_dense_12175my_dense_12177*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_my_dense_layer_call_and_return_conditional_losses_12024j
ReluRelu)my_dense/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:�����������
"my_dense_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0my_dense_1_12181my_dense_1_12183*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_1_layer_call_and_return_conditional_losses_11995n
Relu_1Relu+my_dense_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:�����������
"my_dense_2/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0my_dense_2_12187my_dense_2_12189*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_2_layer_call_and_return_conditional_losses_11966m
Relu_2Relu+my_dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@�
"my_dense_3/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0my_dense_3_12193my_dense_3_12195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_3_layer_call_and_return_conditional_losses_11937m
Relu_3Relu+my_dense_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:��������� �
"my_dense_4/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0my_dense_4_12199my_dense_4_12201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_4_layer_call_and_return_conditional_losses_11908z
IdentityIdentity+my_dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp!^my_dense/StatefulPartitionedCall#^my_dense_1/StatefulPartitionedCall#^my_dense_2/StatefulPartitionedCall#^my_dense_3/StatefulPartitionedCall#^my_dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2D
 my_dense/StatefulPartitionedCall my_dense/StatefulPartitionedCall2H
"my_dense_1/StatefulPartitionedCall"my_dense_1/StatefulPartitionedCall2H
"my_dense_2/StatefulPartitionedCall"my_dense_2/StatefulPartitionedCall2H
"my_dense_3/StatefulPartitionedCall"my_dense_3/StatefulPartitionedCall2H
"my_dense_4/StatefulPartitionedCall"my_dense_4/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
*__inference_my_dense_3_layer_call_fn_12496

inputs
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_3_layer_call_and_return_conditional_losses_11937o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
*__inference_my_dense_1_layer_call_fn_12420

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_1_layer_call_and_return_conditional_losses_11995p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_my_model_layer_call_and_return_conditional_losses_12091

inputs"
my_dense_12061:
��
my_dense_12063:	�$
my_dense_1_12067:
��
my_dense_1_12069:	�#
my_dense_2_12073:	�@
my_dense_2_12075:@"
my_dense_3_12079:@ 
my_dense_3_12081: "
my_dense_4_12085: 

my_dense_4_12087:

identity�� my_dense/StatefulPartitionedCall�"my_dense_1/StatefulPartitionedCall�"my_dense_2/StatefulPartitionedCall�"my_dense_3/StatefulPartitionedCall�"my_dense_4/StatefulPartitionedCall�
 my_dense/StatefulPartitionedCallStatefulPartitionedCallinputsmy_dense_12061my_dense_12063*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_my_dense_layer_call_and_return_conditional_losses_12024j
ReluRelu)my_dense/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:�����������
"my_dense_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0my_dense_1_12067my_dense_1_12069*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_1_layer_call_and_return_conditional_losses_11995n
Relu_1Relu+my_dense_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:�����������
"my_dense_2/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0my_dense_2_12073my_dense_2_12075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_2_layer_call_and_return_conditional_losses_11966m
Relu_2Relu+my_dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������@�
"my_dense_3/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0my_dense_3_12079my_dense_3_12081*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_3_layer_call_and_return_conditional_losses_11937m
Relu_3Relu+my_dense_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:��������� �
"my_dense_4/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0my_dense_4_12085my_dense_4_12087*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_4_layer_call_and_return_conditional_losses_11908z
IdentityIdentity+my_dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp!^my_dense/StatefulPartitionedCall#^my_dense_1/StatefulPartitionedCall#^my_dense_2/StatefulPartitionedCall#^my_dense_3/StatefulPartitionedCall#^my_dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2D
 my_dense/StatefulPartitionedCall my_dense/StatefulPartitionedCall2H
"my_dense_1/StatefulPartitionedCall"my_dense_1/StatefulPartitionedCall2H
"my_dense_2/StatefulPartitionedCall"my_dense_2/StatefulPartitionedCall2H
"my_dense_3/StatefulPartitionedCall"my_dense_3/StatefulPartitionedCall2H
"my_dense_4/StatefulPartitionedCall"my_dense_4/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_my_dense_layer_call_fn_12382

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_my_dense_layer_call_and_return_conditional_losses_12024p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
C__inference_my_dense_layer_call_and_return_conditional_losses_12024

inputs2
matmul_readvariableop_resource:
��*
add_readvariableop_resource:	�
identity��add/ReadVariableOp�matmul/ReadVariableOpv
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0m
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
IdentityIdentityadd:z:0^NoOp*
T0*(
_output_shapes
:����������s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_3_layer_call_and_return_conditional_losses_12506

inputs0
matmul_readvariableop_resource:@ )
add_readvariableop_resource: 
identity��add/ReadVariableOp�matmul/ReadVariableOpt
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0l
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:��������� s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
*__inference_my_dense_1_layer_call_fn_12411

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_1_layer_call_and_return_conditional_losses_11805p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_2_layer_call_and_return_conditional_losses_11966

inputs1
matmul_readvariableop_resource:	�@)
add_readvariableop_resource:@
identity��add/ReadVariableOp�matmul/ReadVariableOpu
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0l
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������@s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_3_layer_call_and_return_conditional_losses_11839

inputs0
matmul_readvariableop_resource:@ )
add_readvariableop_resource: 
identity��add/ReadVariableOp�matmul/ReadVariableOpt
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0l
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:��������� s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_4_layer_call_and_return_conditional_losses_11908

inputs0
matmul_readvariableop_resource: 
)
add_readvariableop_resource:

identity��add/ReadVariableOp�matmul/ReadVariableOpt
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype0i
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:
*
dtype0l
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������
s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_1_layer_call_and_return_conditional_losses_11995

inputs2
matmul_readvariableop_resource:
��*
add_readvariableop_resource:	�
identity��add/ReadVariableOp�matmul/ReadVariableOpv
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0m
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
IdentityIdentityadd:z:0^NoOp*
T0*(
_output_shapes
:����������s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_my_dense_2_layer_call_fn_12458

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_dense_2_layer_call_and_return_conditional_losses_11966o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_my_dense_3_layer_call_and_return_conditional_losses_11937

inputs0
matmul_readvariableop_resource:@ )
add_readvariableop_resource: 
identity��add/ReadVariableOp�matmul/ReadVariableOpt
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0l
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:��������� s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
C__inference_my_dense_layer_call_and_return_conditional_losses_12402

inputs2
matmul_readvariableop_resource:
��*
add_readvariableop_resource:	�
identity��add/ReadVariableOp�matmul/ReadVariableOpv
matmul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
matmulMatMulinputsmatmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:�*
dtype0m
addAddV2matmul:product:0add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
IdentityIdentityadd:z:0^NoOp*
T0*(
_output_shapes
:����������s
NoOpNoOp^add/ReadVariableOp^matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2(
add/ReadVariableOpadd/ReadVariableOp2.
matmul/ReadVariableOpmatmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�+
�
C__inference_my_model_layer_call_and_return_conditional_losses_12364

inputs;
'my_dense_matmul_readvariableop_resource:
��3
$my_dense_add_readvariableop_resource:	�=
)my_dense_1_matmul_readvariableop_resource:
��5
&my_dense_1_add_readvariableop_resource:	�<
)my_dense_2_matmul_readvariableop_resource:	�@4
&my_dense_2_add_readvariableop_resource:@;
)my_dense_3_matmul_readvariableop_resource:@ 4
&my_dense_3_add_readvariableop_resource: ;
)my_dense_4_matmul_readvariableop_resource: 
4
&my_dense_4_add_readvariableop_resource:

identity��my_dense/add/ReadVariableOp�my_dense/matmul/ReadVariableOp�my_dense_1/add/ReadVariableOp� my_dense_1/matmul/ReadVariableOp�my_dense_2/add/ReadVariableOp� my_dense_2/matmul/ReadVariableOp�my_dense_3/add/ReadVariableOp� my_dense_3/matmul/ReadVariableOp�my_dense_4/add/ReadVariableOp� my_dense_4/matmul/ReadVariableOp�
my_dense/matmul/ReadVariableOpReadVariableOp'my_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0|
my_dense/matmulMatMulinputs&my_dense/matmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
my_dense/add/ReadVariableOpReadVariableOp$my_dense_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
my_dense/addAddV2my_dense/matmul:product:0#my_dense/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluRelumy_dense/add:z:0*
T0*(
_output_shapes
:�����������
 my_dense_1/matmul/ReadVariableOpReadVariableOp)my_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
my_dense_1/matmulMatMulRelu:activations:0(my_dense_1/matmul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
my_dense_1/add/ReadVariableOpReadVariableOp&my_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
my_dense_1/addAddV2my_dense_1/matmul:product:0%my_dense_1/add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������U
Relu_1Relumy_dense_1/add:z:0*
T0*(
_output_shapes
:�����������
 my_dense_2/matmul/ReadVariableOpReadVariableOp)my_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
my_dense_2/matmulMatMulRelu_1:activations:0(my_dense_2/matmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
my_dense_2/add/ReadVariableOpReadVariableOp&my_dense_2_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
my_dense_2/addAddV2my_dense_2/matmul:product:0%my_dense_2/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@T
Relu_2Relumy_dense_2/add:z:0*
T0*'
_output_shapes
:���������@�
 my_dense_3/matmul/ReadVariableOpReadVariableOp)my_dense_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
my_dense_3/matmulMatMulRelu_2:activations:0(my_dense_3/matmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
my_dense_3/add/ReadVariableOpReadVariableOp&my_dense_3_add_readvariableop_resource*
_output_shapes
: *
dtype0�
my_dense_3/addAddV2my_dense_3/matmul:product:0%my_dense_3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� T
Relu_3Relumy_dense_3/add:z:0*
T0*'
_output_shapes
:��������� �
 my_dense_4/matmul/ReadVariableOpReadVariableOp)my_dense_4_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype0�
my_dense_4/matmulMatMulRelu_3:activations:0(my_dense_4/matmul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
my_dense_4/add/ReadVariableOpReadVariableOp&my_dense_4_add_readvariableop_resource*
_output_shapes
:
*
dtype0�
my_dense_4/addAddV2my_dense_4/matmul:product:0%my_dense_4/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
a
IdentityIdentitymy_dense_4/add:z:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^my_dense/add/ReadVariableOp^my_dense/matmul/ReadVariableOp^my_dense_1/add/ReadVariableOp!^my_dense_1/matmul/ReadVariableOp^my_dense_2/add/ReadVariableOp!^my_dense_2/matmul/ReadVariableOp^my_dense_3/add/ReadVariableOp!^my_dense_3/matmul/ReadVariableOp^my_dense_4/add/ReadVariableOp!^my_dense_4/matmul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : : : 2:
my_dense/add/ReadVariableOpmy_dense/add/ReadVariableOp2@
my_dense/matmul/ReadVariableOpmy_dense/matmul/ReadVariableOp2>
my_dense_1/add/ReadVariableOpmy_dense_1/add/ReadVariableOp2D
 my_dense_1/matmul/ReadVariableOp my_dense_1/matmul/ReadVariableOp2>
my_dense_2/add/ReadVariableOpmy_dense_2/add/ReadVariableOp2D
 my_dense_2/matmul/ReadVariableOp my_dense_2/matmul/ReadVariableOp2>
my_dense_3/add/ReadVariableOpmy_dense_3/add/ReadVariableOp2D
 my_dense_3/matmul/ReadVariableOp my_dense_3/matmul/ReadVariableOp2>
my_dense_4/add/ReadVariableOpmy_dense_4/add/ReadVariableOp2D
 my_dense_4/matmul/ReadVariableOp my_dense_4/matmul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
<
input_11
serving_default_input_1:0����������<
output_10
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

fc3
fc4
fc5
	optimizer

signatures"
_tf_keras_model
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_1
 trace_2
!trace_32�
(__inference_my_model_layer_call_fn_11886
(__inference_my_model_layer_call_fn_12263
(__inference_my_model_layer_call_fn_12288
(__inference_my_model_layer_call_fn_12139�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1z trace_2z!trace_3
�
"trace_0
#trace_1
$trace_2
%trace_32�
C__inference_my_model_layer_call_and_return_conditional_losses_12326
C__inference_my_model_layer_call_and_return_conditional_losses_12364
C__inference_my_model_layer_call_and_return_conditional_losses_12172
C__inference_my_model_layer_call_and_return_conditional_losses_12205�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z"trace_0z#trace_1z$trace_2z%trace_3
�B�
 __inference__wrapped_model_11771input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
w

kernal
b
bias"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
w

kernal
b
bias"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
w

kernal
b
bias"
_tf_keras_layer
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
w

kernal
b
bias"
_tf_keras_layer
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
w

kernal
b
bias"
_tf_keras_layer
�
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_ratem�m�m�m�m�m�m�m�m�m�v�v�v�v�v�v�v�v�v�v�"
	optimizer
,
Iserving_default"
signature_map
:
��2w
:�2b
:
��2w
:�2b
:	�@2w
:@2b
:@ 2w
: 2b
: 
2w
:
2b
 "
trackable_list_wrapper
C
0
	1

2
3
4"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_my_model_layer_call_fn_11886input_1"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_my_model_layer_call_fn_12263inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_my_model_layer_call_fn_12288inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_my_model_layer_call_fn_12139input_1"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_my_model_layer_call_and_return_conditional_losses_12326inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_my_model_layer_call_and_return_conditional_losses_12364inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_my_model_layer_call_and_return_conditional_losses_12172input_1"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_my_model_layer_call_and_return_conditional_losses_12205input_1"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
�
Qtrace_0
Rtrace_12�
(__inference_my_dense_layer_call_fn_12373
(__inference_my_dense_layer_call_fn_12382�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQtrace_0zRtrace_1
�
Strace_0
Ttrace_12�
C__inference_my_dense_layer_call_and_return_conditional_losses_12392
C__inference_my_dense_layer_call_and_return_conditional_losses_12402�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zStrace_0zTtrace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
Ztrace_0
[trace_12�
*__inference_my_dense_1_layer_call_fn_12411
*__inference_my_dense_1_layer_call_fn_12420�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0z[trace_1
�
\trace_0
]trace_12�
E__inference_my_dense_1_layer_call_and_return_conditional_losses_12430
E__inference_my_dense_1_layer_call_and_return_conditional_losses_12440�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z\trace_0z]trace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
ctrace_0
dtrace_12�
*__inference_my_dense_2_layer_call_fn_12449
*__inference_my_dense_2_layer_call_fn_12458�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zctrace_0zdtrace_1
�
etrace_0
ftrace_12�
E__inference_my_dense_2_layer_call_and_return_conditional_losses_12468
E__inference_my_dense_2_layer_call_and_return_conditional_losses_12478�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zetrace_0zftrace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
ltrace_0
mtrace_12�
*__inference_my_dense_3_layer_call_fn_12487
*__inference_my_dense_3_layer_call_fn_12496�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zltrace_0zmtrace_1
�
ntrace_0
otrace_12�
E__inference_my_dense_3_layer_call_and_return_conditional_losses_12506
E__inference_my_dense_3_layer_call_and_return_conditional_losses_12516�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zntrace_0zotrace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
�
utrace_0
vtrace_12�
*__inference_my_dense_4_layer_call_fn_12525
*__inference_my_dense_4_layer_call_fn_12534�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zutrace_0zvtrace_1
�
wtrace_0
xtrace_12�
E__inference_my_dense_4_layer_call_and_return_conditional_losses_12544
E__inference_my_dense_4_layer_call_and_return_conditional_losses_12554�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zwtrace_0zxtrace_1
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
#__inference_signature_wrapper_12238input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
y	variables
z	keras_api
	{total
	|count"
_tf_keras_metric
`
}	variables
~	keras_api
	total

�count
�
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_my_dense_layer_call_fn_12373inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_my_dense_layer_call_fn_12382inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_my_dense_layer_call_and_return_conditional_losses_12392inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_my_dense_layer_call_and_return_conditional_losses_12402inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_my_dense_1_layer_call_fn_12411inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_my_dense_1_layer_call_fn_12420inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_my_dense_1_layer_call_and_return_conditional_losses_12430inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_my_dense_1_layer_call_and_return_conditional_losses_12440inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_my_dense_2_layer_call_fn_12449inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_my_dense_2_layer_call_fn_12458inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_my_dense_2_layer_call_and_return_conditional_losses_12468inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_my_dense_2_layer_call_and_return_conditional_losses_12478inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_my_dense_3_layer_call_fn_12487inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_my_dense_3_layer_call_fn_12496inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_my_dense_3_layer_call_and_return_conditional_losses_12506inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_my_dense_3_layer_call_and_return_conditional_losses_12516inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_my_dense_4_layer_call_fn_12525inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_my_dense_4_layer_call_fn_12534inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_my_dense_4_layer_call_and_return_conditional_losses_12544inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_my_dense_4_layer_call_and_return_conditional_losses_12554inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
{0
|1"
trackable_list_wrapper
-
y	variables"
_generic_user_object
:  (2total
:  (2count
/
0
�1"
trackable_list_wrapper
-
}	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
:
��2Adam/w/m
:�2Adam/b/m
:
��2Adam/w/m
:�2Adam/b/m
:	�@2Adam/w/m
:@2Adam/b/m
:@ 2Adam/w/m
: 2Adam/b/m
: 
2Adam/w/m
:
2Adam/b/m
:
��2Adam/w/v
:�2Adam/b/v
:
��2Adam/w/v
:�2Adam/b/v
:	�@2Adam/w/v
:@2Adam/b/v
:@ 2Adam/w/v
: 2Adam/b/v
: 
2Adam/w/v
:
2Adam/b/v�
 __inference__wrapped_model_11771t
1�.
'�$
"�
input_1����������
� "3�0
.
output_1"�
output_1���������
�
E__inference_my_dense_1_layer_call_and_return_conditional_losses_12430i4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
E__inference_my_dense_1_layer_call_and_return_conditional_losses_12440i4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
*__inference_my_dense_1_layer_call_fn_12411^4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
*__inference_my_dense_1_layer_call_fn_12420^4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
E__inference_my_dense_2_layer_call_and_return_conditional_losses_12468h4�1
*�'
!�
inputs����������
p 
� ",�)
"�
tensor_0���������@
� �
E__inference_my_dense_2_layer_call_and_return_conditional_losses_12478h4�1
*�'
!�
inputs����������
p
� ",�)
"�
tensor_0���������@
� �
*__inference_my_dense_2_layer_call_fn_12449]4�1
*�'
!�
inputs����������
p 
� "!�
unknown���������@�
*__inference_my_dense_2_layer_call_fn_12458]4�1
*�'
!�
inputs����������
p
� "!�
unknown���������@�
E__inference_my_dense_3_layer_call_and_return_conditional_losses_12506g3�0
)�&
 �
inputs���������@
p 
� ",�)
"�
tensor_0��������� 
� �
E__inference_my_dense_3_layer_call_and_return_conditional_losses_12516g3�0
)�&
 �
inputs���������@
p
� ",�)
"�
tensor_0��������� 
� �
*__inference_my_dense_3_layer_call_fn_12487\3�0
)�&
 �
inputs���������@
p 
� "!�
unknown��������� �
*__inference_my_dense_3_layer_call_fn_12496\3�0
)�&
 �
inputs���������@
p
� "!�
unknown��������� �
E__inference_my_dense_4_layer_call_and_return_conditional_losses_12544g3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0���������

� �
E__inference_my_dense_4_layer_call_and_return_conditional_losses_12554g3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0���������

� �
*__inference_my_dense_4_layer_call_fn_12525\3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown���������
�
*__inference_my_dense_4_layer_call_fn_12534\3�0
)�&
 �
inputs��������� 
p
� "!�
unknown���������
�
C__inference_my_dense_layer_call_and_return_conditional_losses_12392i4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
C__inference_my_dense_layer_call_and_return_conditional_losses_12402i4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
(__inference_my_dense_layer_call_fn_12373^4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
(__inference_my_dense_layer_call_fn_12382^4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
C__inference_my_model_layer_call_and_return_conditional_losses_12172q
5�2
+�(
"�
input_1����������
p 
� ",�)
"�
tensor_0���������

� �
C__inference_my_model_layer_call_and_return_conditional_losses_12205q
5�2
+�(
"�
input_1����������
p
� ",�)
"�
tensor_0���������

� �
C__inference_my_model_layer_call_and_return_conditional_losses_12326p
4�1
*�'
!�
inputs����������
p 
� ",�)
"�
tensor_0���������

� �
C__inference_my_model_layer_call_and_return_conditional_losses_12364p
4�1
*�'
!�
inputs����������
p
� ",�)
"�
tensor_0���������

� �
(__inference_my_model_layer_call_fn_11886f
5�2
+�(
"�
input_1����������
p 
� "!�
unknown���������
�
(__inference_my_model_layer_call_fn_12139f
5�2
+�(
"�
input_1����������
p
� "!�
unknown���������
�
(__inference_my_model_layer_call_fn_12263e
4�1
*�'
!�
inputs����������
p 
� "!�
unknown���������
�
(__inference_my_model_layer_call_fn_12288e
4�1
*�'
!�
inputs����������
p
� "!�
unknown���������
�
#__inference_signature_wrapper_12238
<�9
� 
2�/
-
input_1"�
input_1����������"3�0
.
output_1"�
output_1���������

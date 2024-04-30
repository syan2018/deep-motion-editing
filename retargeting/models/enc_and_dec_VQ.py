import torch
import torch.nn as nn
from models.skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear

from .tools.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
class Encoder(nn.Module):
    """
    Encoder类用于构建编码器模型。

    参数:
    - args: 包含模型配置的参数对象，例如旋转类型、卷积核大小、层数等。
    - topology: 骨架结构，表示节点和边的关系。

    属性:
    - topologies: 存储每一层的骨架结构。
    - channel_base: 存储每一层的基础通道数。
    - channel_list: 存储每一层的输入输出通道数。
    - edge_num: 存储每一层的边数。
    - pooling_list: 存储每一层的池化操作列表。
    - layers: 存储所有层的模块列表。
    - convs: 存储所有卷积层的列表。
    - last_channel: 最后一层的通道数。
    """

    def __init__(self, args, topology):
        super(Encoder, self).__init__()
        self.topologies = [topology]
        # 根据旋转类型确定基础通道数
        if args.rotation == 'euler_angle': self.channel_base = [3]
        elif args.rotation == 'quaternion': self.channel_base = [4]
        self.channel_list = []
        self.edge_num = [len(topology) + 1]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        self.convs = []

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2
        bias = True
        # 根据骨架信息确定是否添加偏移量
        if args.skeleton_info == 'concat': add_offset = True
        else: add_offset = False

        # 计算每一层的通道数
        for i in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)

        # 构建卷积层和池化层
        for i in range(args.num_layers):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i+1] * self.edge_num[i]
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            # 添加额外的卷积层
            for _ in range(args.extra_conv):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
                                        padding=padding, padding_mode=args.padding_mode, bias=bias))
            # 添加主卷积层和偏移量添加模块（如果需要）
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                    padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
            self.convs.append(seq[-1])
            last_pool = True if i == args.num_layers - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]) + 1)
            if i == args.num_layers - 1:
                self.last_channel = self.edge_num[-1] * self.channel_base[i + 1]

    def forward(self, input, offset=None):
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
            input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)

        for i, layer in enumerate(self.layers):
            if self.args.skeleton_info == 'concat' and offset is not None:
                self.convs[i].set_offset(offset[i])
            input = layer(input)
        return input


class Decoder(nn.Module):
    def __init__(self, args, enc: Encoder):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.args = args
        self.enc = enc
        self.convs = []

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2

        if args.skeleton_info == 'concat': add_offset = True
        else: add_offset = False

        for i in range(args.num_layers):
            seq = []
            in_channels = enc.channel_list[args.num_layers - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[args.num_layers - i - 1], args.skeleton_dist)

            if i != 0 and i != args.num_layers - 1:
                bias = False
            else:
                bias = True

            self.unpools.append(SkeletonUnpool(enc.pooling_list[args.num_layers - i - 1], in_channels // len(neighbor_list)))

            seq.append(nn.Upsample(scale_factor=2, mode=args.upsampling, align_corners=False))
            seq.append(self.unpools[-1])
            for _ in range(args.extra_conv):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=enc.edge_num[args.num_layers - i - 1], kernel_size=kernel_size,
                                        stride=1,
                                        padding=padding, padding_mode=args.padding_mode, bias=bias))
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=enc.edge_num[args.num_layers - i - 1], kernel_size=kernel_size, stride=1,
                                    padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * enc.channel_base[args.num_layers - i - 1] // enc.channel_base[0]))
            self.convs.append(seq[-1])
            if i != args.num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, input, offset=None):
        for i, layer in enumerate(self.layers):
            if self.args.skeleton_info == 'concat':
                self.convs[i].set_offset(offset[len(self.layers) - i - 1])
            input = layer(input)
        # throw the padded rwo for global position
        if self.args.rotation == 'quaternion' and self.args.pos_repr != '4d':
            input = input[:, :-1, :]

        return input



class VQVAE(nn.Module):
    def __init__(self, args, topology,
                 quantizer: nn.Module,
                 ):
        super(VQVAE, self).__init__()

        # code_num 为码本大小
        self.code_num = args.code_num
        # code_dim 为输入编码维度
        channel_base = 4 if args.rotation == 'quaternion' else 3
        self.code_dim = 7 * channel_base * 2 ** args.num_layers
        self.quantizer = quantizer

        self.enc = Encoder(args, topology)
        self.dec = Decoder(args, self.enc)


    def forward(self, input, offset=None):
        latent = self.enc(input, offset)
        # latent = (N, C, L) Batch Size, Channel Size, Seq Length
        N, C, L = latent.shape
      
        x_d, loss, perplexity, code_idx = self.quantizer(latent)

        code_idx = code_idx.view(N, L)
        # 然后，在第二个维度插入一个大小为 1 的新维度，以得到最终形状 [256, 1, 16]
        code_idx = code_idx.unsqueeze(1)

        # print("-------- print latent and code shape -------")
        # print(latent.shape)     # torch.Size([256, 112, 16])
        # print(code_idx.shape)   # torch.Size([4096])  == torch.Size([256, 16])
        # print("---------------")

        result = self.dec(x_d, offset)
        return code_idx, result, loss, perplexity

    def encode(self, input, offset=None):
        """
        将标准动作转换为编码向量
        输入维度为 (N, C, L)
        输出维度为 (N * L)
        (由于Encoder，会有缩短时间步长的情况)
        """
        #
        latent = self.enc(input, offset)
        code_idx = self.quantizer.quantize(latent)

        return code_idx

    def decode(self, z, offset=None):

        x_d = self.quantizer.dequantize(z)
        pos = self.dec(x_d, offset)

        return pos





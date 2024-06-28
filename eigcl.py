# -*- coding: utf-8 -*-
# @Time   : 2023/11/6
# @Author : Lingzhen Zhou
# @Email  : a15840732040@gmail.com
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pandas as pd
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

class EIGCL(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(EIGCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.group_user_matrix = self.load_group_matrix(dataset.dataset_name)
        self.n_neighbors = config["n_neighbors"]
        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]
        self.ssl_temp = config["ssl_temp"]
        self.embedding_size = config["embedding_size"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]
        self.ssl_reg = config["ssl_reg"]
        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.uu_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.ii_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.groupu_embedding = torch.nn.Embedding(
            num_embeddings=self.n_groups, embedding_dim=self.latent_dim
        )
        self.groupi_embedding = torch.nn.Embedding(
            num_embeddings=self.n_groups, embedding_dim=self.latent_dim
        )
        self.ugroup_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.igroup_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.restore_group_e = None

        # generate intermediate data
        self.norm_uiadj_matrix = self.get_norm_uiadj_mat().to(self.device)
        self.norm_uuadj_matrix = self.get_norm_uuadj_mat().to(self.device)
        self.norm_iiadj_matrix = self.get_norm_iiadj_mat().to(self.device)
        self.norm_guadj_matrix = self.get_norm_guadj_mat().to(self.device)
        self.norm_giadj_matrix = self.get_norm_giadj_mat().to(self.device)
        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e", "restore_group_e"]

    def load_group_matrix(self, dataset):
        r"""读取用户-群组关系数据

        输入：数据集名称
        输出：用户-群组关系矩阵
        """
        data_path = os.path.join('dataset/', f"{dataset}/{dataset}.net")
        df = pd.read_csv(
            data_path,
            delimiter='\t',
            encoding="utf-8",
            usecols=[0, 1],
            engine="python",
        )
        df.columns = ["user", "group"]
        df.dropna()
        df.drop_duplicates()
        n_groups = df.max()["group"]
        self.n_groups = n_groups + 1
        row = df["group"].tolist()
        col = df["user"].tolist()
        val = [1] * len(row)
        group_matrix = sp.coo_matrix((val, (row, col)), shape=(self.n_groups, self.n_users), dtype=np.float32)
        # for idx, data in df.iterrows():
        #     group_matrix[data["group"], data["user"]] = 1
        return group_matrix

    def get_norm_guadj_mat(self):
        r"""Get the normalized interaction matrix of users and groups.

                Construct the square matrix from the training data and normalize it
                using the laplace matrix.

                .. math::
                    A_{hat} = D^{-0.5} \times A \times D^{-0.5}

                Returns:
                    Sparse tensor of the normalized interaction matrix.
        """
        # build g-u adj matrix
        A = sp.dok_matrix(
            (self.n_groups + self.n_users, self.n_groups + self.n_users), dtype=np.float32
        )
        inter_M = self.group_user_matrix
        inter_M_t = self.group_user_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_groups), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_groups, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_norm_giadj_mat(self):
        r"""Get the normalized interaction matrix of items and groups.

                Construct the square matrix from the training data and normalize it
                using the laplace matrix.

                .. math::
                    A_{hat} = D^{-0.5} \times A \times D^{-0.5}

                Returns:
                    Sparse tensor of the normalized interaction matrix.
        """
        # build g-u adj matrix
        A = sp.dok_matrix(
            (self.n_groups + self.n_items, self.n_groups + self.n_items), dtype=np.float32
        )
        inter_M = self.group_user_matrix.dot(self.interaction_matrix)
        inter_M_gi = sp.dok_matrix((self.n_groups, self.n_items), dtype=np.float32)
        for i in range(self.n_groups):
            row = torch.from_numpy(inter_M.getrow(i).toarray()[0])
            row_sims, row_idxs = torch.topk(row, self.n_neighbors)
            for j in range(self.n_neighbors):
                inter_M_gi[i, row_idxs[j]] = row_sims[j]
        inter_M_gi = inter_M_gi.tocoo()
        inter_M_t = inter_M_gi.transpose()
        data_dict = dict(
            zip(zip(inter_M_gi.row, inter_M_gi.col + self.n_groups), [1] * inter_M_gi.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_groups, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = A.sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_norm_uiadj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build u-i adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning 
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_norm_uuadj_mat(self):
        r"""Get the normalized interaction matrix of users and users.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build u-u adj matrix
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        A = inter_M.dot(inter_M_t)
        A[range(self.n_users), range(self.n_users)] = 0
        inter_M_uu = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)
        for i in range(self.n_users):
            row = torch.from_numpy(A.getrow(i).toarray()[0])
            row_sims, row_idxs = torch.topk(row, self.n_neighbors)
            for j in range(self.n_neighbors):
                inter_M_uu[i, row_idxs[j]] = row_sims[j]
        A = inter_M_uu
        # norm adj matrix
        sumArr = A.sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_norm_iiadj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build u-u adj matrix
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        A = inter_M_t.dot(inter_M)
        A[range(self.n_items), range(self.n_items)] = 0
        inter_M_ii = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        for i in range(self.n_items):
            row = torch.from_numpy(A.getrow(i).toarray()[0])
            row_sims, row_idxs = torch.topk(row, self.n_neighbors)
            for j in range(self.n_neighbors):
                inter_M_ii[i, row_idxs[j]] = row_sims[j]
        A = inter_M_ii
        # norm adj matrix
        sumArr = A.sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        gu_embeddings = self.groupu_embedding.weight
        gi_embeddings = self.groupi_embedding.weight
        ug_embeddings = self.ugroup_embedding.weight
        ig_embeddings = self.igroup_embedding.weight
        ego_embeddings_ui = torch.cat([user_embeddings, item_embeddings], dim=0)
        ego_embeddings_uu = self.uu_embedding.weight
        ego_embeddings_ii = self.ii_embedding.weight
        ego_embeddings_gu = torch.cat([gu_embeddings, ug_embeddings], dim=0)
        ego_embeddings_gi = torch.cat([gi_embeddings, ig_embeddings], dim=0)

        return ego_embeddings_ui, ego_embeddings_uu, ego_embeddings_ii, ego_embeddings_gu, ego_embeddings_gi

    def forward(self):
        all_embeddings, uu_embeddings, ii_embeddings, gu_embeddings, gi_embeddings = self.get_ego_embeddings()
        embeddings_list_ui, embeddings_list_uu, embeddings_list_ii, embeddings_list_gu, embeddings_list_gi = [all_embeddings], [uu_embeddings], [ii_embeddings], [gu_embeddings], [gi_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_uiadj_matrix, all_embeddings)
            embeddings_list_ui.append(all_embeddings)
            uu_embeddings = torch.sparse.mm(self.norm_uuadj_matrix, uu_embeddings)
            embeddings_list_uu.append(uu_embeddings)
            ii_embeddings = torch.sparse.mm(self.norm_iiadj_matrix, ii_embeddings)
            embeddings_list_ii.append(ii_embeddings)
            gu_embeddings = torch.sparse.mm(self.norm_guadj_matrix, gu_embeddings)
            embeddings_list_gu.append(gu_embeddings)
            gi_embeddings = torch.sparse.mm(self.norm_giadj_matrix, gi_embeddings)
            embeddings_list_gi.append(gi_embeddings)

        mymodel_all_embeddings = torch.stack(embeddings_list_ui, dim=1)
        mymodel_all_embeddings = torch.mean(mymodel_all_embeddings, dim=1)
        mymodel_uu_embeddings = torch.stack(embeddings_list_uu, dim=1)
        mymodel_uu_embeddings = torch.mean(mymodel_uu_embeddings, dim=1)
        mymodel_ii_embeddings = torch.stack(embeddings_list_ii, dim=1)
        mymodel_ii_embeddings = torch.mean(mymodel_ii_embeddings, dim=1)
        mymodel_gu_embeddings = torch.stack(embeddings_list_gu, dim=1)
        mymodel_gu_embeddings = torch.mean(mymodel_gu_embeddings, dim=1)
        mymodel_gi_embeddings = torch.stack(embeddings_list_gi, dim=1)
        mymodel_gi_embeddings = torch.mean(mymodel_gi_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            mymodel_all_embeddings, [self.n_users, self.n_items]
        )
        group_gu_embeddings, user_gu_embeddings = torch.split(
            mymodel_gu_embeddings, [self.n_groups, self.n_users]
        )
        group_gi_embeddings, item_gi_embeddings = torch.split(
            mymodel_gi_embeddings, [self.n_groups, self.n_items]
        )
        user_embeddings = torch.cat([user_all_embeddings, mymodel_uu_embeddings, user_gu_embeddings], dim=1)
        item_embeddings = torch.cat([item_all_embeddings, mymodel_ii_embeddings, item_gi_embeddings], dim=1)
        group_embeddings = torch.cat([group_gu_embeddings, group_gi_embeddings])

        return user_embeddings, item_embeddings, group_embeddings

    def ssl_layer_loss(self, current_embedding, previous_embedding, group_embedding, user, item, group):
        current_user_embeddings, current_item_embeddings = torch.split(
            current_embedding, [self.n_users, self.n_items]
        )
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(
            previous_embedding, [self.n_users, self.n_items]
        )
        current_group_embeddings, previous_group_embeddings_all = torch.split(
            group_embedding, [self.n_groups, self.n_groups]
        )
        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        current_group_embeddings = current_group_embeddings[group]
        previous_group_embeddings = previous_group_embeddings_all[group]
        norm_group_emb1 = F.normalize(current_group_embeddings)
        norm_group_emb2 = F.normalize(previous_group_embeddings)
        norm_all_group_emb = F.normalize(previous_group_embeddings_all)
        pos_score_group = torch.mul(norm_group_emb1, norm_group_emb2).sum(dim=1)
        ttl_score_group = torch.matmul(norm_group_emb1, norm_all_group_emb.transpose(0, 1))
        pos_score_group = torch.exp(pos_score_group / self.ssl_temp)
        ttl_score_group = torch.exp(ttl_score_group / self.ssl_temp).sum(dim=1)

        ssl_loss_group = -torch.log(pos_score_group / ttl_score_group).sum()

        ssl_loss = self.ssl_reg * (self.alpha * ssl_loss_user + self.beta * ssl_loss_item + self.gamma * ssl_loss_group)
        return ssl_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        group = torch.LongTensor(range(self.n_groups)).to(self.device)

        user_all_embeddings, item_all_embeddings, group_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        uu_ego_embeddings = self.uu_embedding(user)
        iipos_ego_embeddings = self.ii_embedding(pos_item)
        iineg_ego_embeddings = self.ii_embedding(neg_item)
        gu_ego_embeddings = self.groupu_embedding(group)
        gi_ego_embeddings = self.groupi_embedding(group)
        ug_ego_embeddings = self.ugroup_embedding(user)
        igpos_ego_embedings = self.igroup_embedding(pos_item)
        igneg_ego_embedings = self.igroup_embedding(neg_item)
        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            uu_ego_embeddings,
            iipos_ego_embeddings,
            iineg_ego_embeddings,
            gu_ego_embeddings,
            gi_ego_embeddings,
            ug_ego_embeddings,
            igpos_ego_embedings,
            igneg_ego_embedings,
            require_pow=self.require_pow,
        )
        current_user_embeddings, previous_user_embeddings, group_user_embeddings = torch.split(
            user_all_embeddings, [self.embedding_size, self.embedding_size, self.embedding_size], dim=1
        )
        current_item_embeddings, previous_item_embeddings, group_item_embeddings = torch.split(
            item_all_embeddings, [self.embedding_size, self.embedding_size, self.embedding_size], dim=1
        )
        ssl_loss = self.ssl_layer_loss(torch.cat([current_user_embeddings, current_item_embeddings]), torch.cat([previous_user_embeddings, previous_item_embeddings]), group_all_embeddings, user, pos_item, group)
        loss = mf_loss + self.reg_weight * reg_loss + ssl_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, group_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, self.restore_group_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

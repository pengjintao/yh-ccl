#include <cmath>
#include <unordered_map>
#include <vector>

struct allreduce_model
{
    int n = 100;
    int I = (1 << 18);
    int p = 1;
    int r = 24;
    double C = 1.0 / (38592.5);             //天河2
    double G = 1.0 / 38369.85;              //天河2
    double l = 2.5;                         //天河2
    double G1 = 23117.685318 / 268435456.0; //天河2双向
    double g = 0.704757;
    double a = 0.1340161875; // 5ns
    // double o = 1; logp中的处理器上开销o被忽略
    long long L = (1 << 27);
    int S;
    int A = 25;
    int B;
    void init()
    {
        S = int(L / (I * p));
        if (L % (I * p) != 0)
            S += 1;
        G = C * 2.0 / 3.0;
        F_NA_map.clear();
        B = n / A + 1;
    }
    void print()
    {
        init();
        printf("节点数量:%d\n", n);
        printf("节点内片大小:%d\n", I);
        printf("节点间片对比节点内片比例:%d\n", r);
        printf("节点间片大小:%d\n", get_I1());
        printf("节点内进程数量:%d\n", p);
        printf("单核心计算时间:%f\n", C);
        printf("节点内LogGP中的带宽的倒数:%f\n", G);
        printf("节点间LogGP中的延迟参数:%f\n", l);
        printf("节点间LogGP中的带宽的倒数:%f\n", G1);
        printf("节点间LogGP中的gap the gap between messages,:%f\n", g);
        printf("一次原子操作的开销:%f\n", a);
        printf("输入消息大小:%lld\n", L);
        printf("总大步数量:%d\n", S);
        // printf("MPI eager协议或者rendezvous 协议大小切换:%d\n", h);
        //天津UCX是8KB
        //天河2 MPICH是 MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE=131072 2**17
        // export MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE=8192
        // printf("RDMA传输或拷贝传输的阈值消息大小:%d\n", h);
    }
    int get_I1() { return int(I * r); }
    int step_msgsz(int i) { return std::min(L - i * I * p, (long long)I * p); }

    double T_NA(int i)
    {
        //完成第i个ia大步的总消息量
        int sz = step_msgsz(i);
        //需要完成的消息量
        int lsz = sz * (A - 1.0) / A + sz * (A - 1.0) * (B - 1.0) / (A * B);
        int I1 = std::min(get_I1(), sz);
        int activate_leadern = (sz / I1);
        if (sz % I1 != 0)
            activate_leadern++;
        //多核心对外通信模型
        //节点间总消息数量
        int msg_ct = activate_leadern * 2 * (A - 1 + B - 1);
        double latencty_o = 2.0 * (A - 1 + B - 1) * l;
        double gap_o = msg_ct * g;
        //节点间产生的
        //统计总数据传数量带来的开销
        double logp_band_o = 2 * lsz * G1;
        //原子操作更新带来的开销
        double atomic_o = a * (1 + sz / (I1));
        // std::cout<<" i= "<<i<<" gap_o= "<<gap_o<<std::endl;
        return latencty_o + gap_o + logp_band_o + atomic_o;
    }
    double F_NA(int i)
    {

        //输入的i代表节点内大步id

        if (F_NA_map.find(i) == F_NA_map.end())
        {
            double re;
            if (i > 0)
            {
                re = std::max(F_R(i), F_NA(i - 1)) + T_NA(i);
            }
            else
            {
                re = F_R(i) + T_NA(i);
            }
            F_NA_map[i] = re;
            return re;
        }
        else
        {
            return F_NA_map[i];
        }
    }

    std::unordered_map<int, double> F_NA_map;

    double T_R(int i)
    {

        int sz = step_msgsz(i);
        // single_sz长度的内存拷贝+single_sz长度的计算*p-1+ p次原子操作
        //  std::cout<<sz*G + (p-1)*sz*C + 2*(p-1)*a<<std::endl;
        return sz * G + (p - 1) * sz * C + 2 * (p - 1) * a;
    }
    double F_R(int i)
    {
        //计算第i个广播大步需要处理的总消息数量
        long long sz;
        if (i == S - 1)
        {
            sz = L;
        }
        else
        {
            sz = (i + 1) * I * p;
        }
        double re = sz * G + (p - 1) * sz * C + 2 * (p - 1) * a * (i + 1);
        F_NA_map[i] = re;
        return re;
    }

    double T_B(int i)
    {
        // std::cout<<"i= "<<i<<" sz= "<<sz<<std::endl;
        double barrierT = 2.0 * p * a;
        int sz = step_msgsz(i);
        int lsz = sz * (A - 1.0) / A + sz * (A - 1.0) * (B - 1.0) / (A * B);
        //统计总节点间计算带来的开销
        double allreduce_calc = C * lsz * (n - 1.0) / n;
        return allreduce_calc + barrierT + p * sz * G;
    }

    double F_B(int i)
    {
        if (n > 1)
        {
            if (i > 0)
            {
                return std::max(F_B(i - 1), F_NA(i)) + T_B(i);
            }
            else
            {

                return std::max(F_R(S - 1), F_NA(i)) + T_B(i);
            }
        }
        else
        {
            if (i > 0)
            {
                return F_B(i - 1) + T_B(i);
            }
            else
            {

                return F_R(S - 1) + T_B(i);
            }
        }
    }

    // double correction_model()
    // {
    //     //输入为 x=[消息大小 [64KB-128MB] L，节点数量 [10-100,10]n，每节点进程数量 [1,2,4,6,8,10,...,24]p，节点内片大小 1<<[16,..,21] I，节点间片比例 [1,2,4,6,8,10,...,24] r,第一层规约节点数量 10]
    //     std::vector<double> x = {L * 1.0, n * 1.0, p * 1.0, I * 1.0, r * 1.0};
    //     int ct = x.size();
    //     //输出为修正值
    //     //修正值加上logGP模型的值要经可能接近真实测量值。

    //     //修正模型1 y = Ax + b
    //     {
    //         std::vector<double> parameters(ct + 1, 0.0);
    //         double re = 0.0;
    //         for (int i = 0; i < ct; i++)
    //             re += x[i] * parameters[i];
    //         re += parameters[ct];
    //         return re;
    //     }
    // }
    double resurvce_loop(int start, int end, int d, int &parameter_i)
    {
        if (d == 0)
            return parameters[parameter_i++];
        if (end - start == 1)
            return X_vec[start][d] * parameters[parameter_i++];
        double re = 0.0;
        for (int di = d; di >= 0; di--)
        {
            re += X_vec[start][di] * resurvce_loop(start + 1, end, d - di, parameter_i);
        }
        return re;
    }
    double linear_product(const std::vector<double> &X)
    {
        double re = intercept;
        for (int i = 0; i < X.size(); i++)
        {
            re += X[i] * parameters[i];
        }
        return re;
    }
    void init_correction(double loggp)
    {
        //第一步对X进行预处理
        //[logGP/(1000*msgsz),ratio,1/ratio,log(msgsz),1/log(msgsz), log(slicesz),1/log(slicesz)]
        std::vector<double> X;
        int len = parameters.size();

        double msgsz = L;
        double noden = n;
        double ppn = p;
        double intra_slicesz = I;
        double ratio = r;
        // printf("%f ", loggp);
        // printf("%f ", msgsz);
        // printf("%f ", noden);
        // printf("%f ", ppn);
        // printf("%f ", intra_slicesz);
        // printf("%f ", ratio);

        X.resize(len);
        X[0] = L / (1000.0 * loggp);
        X[1] = r;
        X[2] = 1.0 / r;
        X[3] = std::log(double(L));
        X[4] = 1.0 / (X[3]);
        // X[5] = std::log(double(I));
        // X[6] = 1.0 / X[5];
        // for (int i = 0; i < 7; i++)
        //     printf("%f ", X[i]);
        // puts("");

        X_vec.resize(degree + 1);
        for (int i = 0; i <= degree; i++)
            X_vec[i].resize(len, 1.0);
        for (int i = 1; i <= degree; i++)
        {
            for (int j = 0; j < len; j++)
            {
                X_vec[i][j] = X_vec[i - 1][j] * X[j];
            }
        }
    }
    int degree = 1;
    std::vector<std::vector<double>> X_vec;

    std::vector<double> parameters = {2.30222765e-01, 6.50450547e-04, -1.09941197e-01,
                                      -1.82032193e-01, -5.17082570e+01};
    double intercept = 6.81120353;
    double loggp;
    double eval()
    {
        init();
        double barrierT = 2.0 * 2.0 * p * a;
        loggp = F_B(S - 1) + barrierT;
        init_correction(loggp);
        int parameter_i = 0;
        double re = linear_product(X_vec[1]);
        // print();
        return re;
        // return F_B(S - 1) + barrierT + correction_model();
    }
};
#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <inttypes.h>
using namespace std;

const int MAX = 10000;

int N, M, n, m;
int matrice[MAX][MAX];
int kernel[5][5];
int results[MAX][MAX];
int results2[MAX][MAX];

void secventialStatic()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            int sum = 0;
            for (int ik = 0; ik < n; ik++)
            {
                for (int jk = 0; jk < m; jk++)
                {
                    int x = i - n / 2 + ik;
                    int y = j - m / 2 + jk;
                    if (x < 0)
                        x = 0;
                    if (y < 0)
                        y = 0;
                    if (x >= N)
                        x = N - 1;
                    if (y >= M)
                        y = M - 1;
                    if (x < 0 && y < 0)
                    {
                        x = 0;
                        y = 0;
                    }
                    if (x < 0 && y >= M)
                    {
                        x = 0;
                        y = M - 1;
                    }
                    if (x >= N && y < 0)
                    {
                        x = N - 1;
                        y = 0;
                    }
                    if (x >= N && y >= M)
                    {
                        x = N - 1;
                        y = M - 1;
                    }
                    sum += matrice[x][y] * kernel[ik][jk];
                }
            }
            results[i][j] = sum;
        }
    }
}

class ThreadLiniiStatic
{
private:
    int start, end;

public:
    ThreadLiniiStatic(int start, int end)
    {
        this->start = start;
        this->end = end;
    }

    void operator()()
    {
        for (int i = start; i < end; i++)
        {
            for (int j = 0; j < M; j++)
            {
                int sum = 0;
                for (int ik = 0; ik < n; ik++)
                {
                    for (int jk = 0; jk < m; jk++)
                    {
                        int x = i - n / 2 + ik;
                        int y = j - m / 2 + jk;
                        if (x < 0)
                            x = 0;
                        if (y < 0)
                            y = 0;
                        if (x >= N)
                            x = N - 1;
                        if (y >= M)
                            y = M - 1;
                        if (x < 0 && y < 0)
                        {
                            x = 0;
                            y = 0;
                        }
                        if (x < 0 && y >= M)
                        {
                            x = 0;
                            y = M - 1;
                        }
                        if (x >= N && y < 0)
                        {
                            x = N - 1;
                            y = 0;
                        }
                        if (x >= N && y >= M)
                        {
                            x = N - 1;
                            y = M - 1;
                        }
                        sum += matrice[x][y] * kernel[ik][jk];
                    }
                }
                results2[i][j] = sum;
            }
        }
    }
};

void convolutiiLiniiStaticParalel(int nr_threads)
{
    thread **threads = new thread *[nr_threads];

    int start = 0;
    int end = 0;

    int liniiPerThread = N / nr_threads;
    int rest = N % nr_threads;

    for (int i = 0; i < nr_threads; i++)
    {
        int linii = liniiPerThread;
        if (rest > 0)
        {
            linii++;
            rest--;
        }
        end = start + linii;
        threads[i] = new thread(ThreadLiniiStatic(start, end));
        start = end;
    }

    for (int i = 0; i < nr_threads; i++)
    {
        threads[i]->join();
        delete threads[i];
    }
    delete[] threads;
}

class ThreadColoaneStatic
{
private:
    int start, end;

public:
    ThreadColoaneStatic(int start, int end)
    {
        this->start = start;
        this->end = end;
    }

    void operator()()
    {
        for (int j = start; j < end; j++)
        {
            for (int i = 0; i < N; i++)
            {
                int sum = 0;
                for (int ik = 0; ik < n; ik++)
                {
                    for (int jk = 0; jk < m; jk++)
                    {
                        int x = i - n / 2 + ik;
                        int y = j - m / 2 + jk;
                        if (x < 0)
                            x = 0;
                        if (y < 0)
                            y = 0;
                        if (x >= N)
                            x = N - 1;
                        if (y >= M)
                            y = M - 1;
                        sum += matrice[x][y] * kernel[ik][jk];
                    }
                }
                results2[i][j] = sum;
            }
        }
    }
};

class ThreadBLock
{
private:
    int starti, endi, startj, endj;

public:
    ThreadBLock(int starti, int endi, int startj, int endj)
    {
        this->starti = starti;
        this->endi = endi;
        this->startj = startj;
        this->endj = endj;
    };
    void operator()()
    {
        for (int i = starti; i < endi; i++)
        {
            int columnStart = (i == starti) ? startj : 0;
            int columnEnd = (i == endi) ? endj : M;
            for (int j = columnStart; j < columnEnd; j++)
            {
                int sum = 0;
                for (int ik = 0; ik < n; ik++)
                {
                    for (int jk = 0; jk < m; jk++)
                    {
                        int x = i - n / 2 + ik;
                        int y = j - m / 2 + jk;
                        if (x < 0)
                            x = 0;
                        if (y < 0)
                            y = 0;
                        if (x >= N)
                            x = N - 1;
                        if (y >= M)
                            y = M - 1;
                        sum += matrice[x][y] * kernel[ik][jk];
                    }
                }
                results2[i][j] = sum;
            }
        }
    }
};

void convolutiiBlockParalelStatic(int nr_threads)
{
    int blocksPerThread = (N * M) / nr_threads;
    int r = (N * M) % nr_threads;
    int startI = 0;
    int startJ = 0;
    int endI = 0;
    int endJ = 0;

    thread **threads = new thread *[nr_threads];

    for (int i = 0; i < nr_threads; i++)
    {
        int blocks = blocksPerThread + (i < r ? 1 : 0);
        while (blocks + startJ >= M)
        {
            endI++;
            blocks = blocks - M + startJ;
            startJ = 0;
        }
        endJ = startJ + blocks;
        threads[i] = new thread(ThreadBLock(startI, endI, startJ, endJ));
        startI = endI;
        startJ = endJ;
        endJ = 0;
    }
    for (int i = 0; i < nr_threads; i++)
    {
        threads[i]->join();
        delete threads[i];
    }
    delete[] threads;
}

void convolutiiColoaneStaticParalel(int nr_threads)
{
    thread **threads_col = new thread *[nr_threads];

    int start = 0;
    int end = 0;

    int coloanePerThread = M / nr_threads;
    int rest = M % nr_threads;

    for (int i = 0; i < nr_threads; i++)
    {
        int coloane = coloanePerThread;
        if (rest > 0)
        {
            coloane++;
            rest--;
        }
        end = start + coloane;
        threads_col[i] = new thread(ThreadColoaneStatic(start, end));
        start = end;
    }

    for (int i = 0; i < nr_threads; i++)
    {
        threads_col[i]->join();
        delete threads_col[i];
    }
    delete[] threads_col;
}

void generateMatrix(int N, int M)
{
    ofstream fout("matrix_kernel_10000_10000_5_5.txt");
    if (!fout)
    {
        cout << "Error opening file for writing. Please check your permissions or if the file is being used by another process." << endl;
        return;
    }

    fout << N << " " << M << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
            fout << rand() % 100 << " ";
        fout << endl;
    }

    fout << 5 << " " << 5 << endl;
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
            fout << rand() % 100 << " ";
        fout << endl;
    }

    fout.close();
}

bool verificaRezultate(int results[][MAX], int result2[][MAX])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
        {
            if (results[i][j] != result2[i][j])
                return false;
        }
    return true;
}

void appendResultsToFile(const std::string &filename, bool areEqual)
{
    std::ofstream fout(filename, std::ios_base::app);
    if (!fout)
    {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }

    fout << "All the files are equal: " << (areEqual ? "True" : "False") << std::endl;
    fout.close();
}

void generate(int N, int M, int n, int m, const std::string &fileName)
{
    std::ofstream writer(fileName);
    if (!writer.is_open())
    {
        std::cerr << "Failed to open file: " << fileName << std::endl;
        return;
    }

    writer << N << " " << M << std::endl;
    std::srand(std::time(nullptr)); // Seed the random number generator

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            writer << std::rand() % 10 << " ";
        }
        writer << std::endl;
    }

    writer << n << " " << m << std::endl;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            writer << std::rand() % 10 << " ";
        }
        writer << std::endl;
    }

    writer.close();
}

int main(int argc, char *argv[])
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (argc != 4)
    {
        std::cerr << "Invalid number of arguments. Please provide the number of threads, the relative path to the file and the orientation." << std::endl;
        return -1;
    }
    int nr_threads = atoi(argv[1]);
    string relative_path = argv[2];
    int orientation = atoi(argv[3]);

    string base_path = "------";
    string path = base_path + relative_path;

    auto start = chrono::high_resolution_clock::now();
    ifstream fin(path);
    fin >> N >> M;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
        {
            fin >> matrice[i][j];
        }
    fin >> n >> m;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            fin >> kernel[i][j];
    fin.close();

    if (orientation == -1)
    {
        auto start = chrono::high_resolution_clock::now();
        secventialStatic();
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
        cout << duration.count() << endl;
    }
    else if (orientation == 0)
    {

        auto start = chrono::high_resolution_clock::now();
        convolutiiLiniiStaticParalel(nr_threads);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
        cout << duration.count() << endl;

        secventialStatic();

        bool areEqual = verificaRezultate(results, results2);
        appendResultsToFile("results.txt", areEqual);
    }
    else if (orientation == 1)
    {
        auto start = chrono::high_resolution_clock::now();
        convolutiiColoaneStaticParalel(nr_threads);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
        cout << duration.count() << endl;

        secventialStatic();
        bool areEqual = verificaRezultate(results, results2);
        appendResultsToFile("results.txt", areEqual);
    }
    else
    {
        auto start = chrono::high_resolution_clock::now();
        convolutiiBlockParalelStatic(nr_threads);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
        cout << duration.count() << endl;

        secventialStatic();
        bool areEqual = verificaRezultate(results, results2);
        appendResultsToFile("results.txt", areEqual);
    }
    return 0;
}
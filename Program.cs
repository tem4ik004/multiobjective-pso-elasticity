using System;
using System.Collections.Generic;
using System.Linq;

namespace MultiObjectivePso
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // ---------- Настройки задачи ----------
            int dimensions = 2;    // x и y
            int objectives = 1;    // одна целевая функция f(x, y) = x^2 + y^2

            double[] lowerBounds = { -10.0, -10.0 };
            double[] upperBounds = {  10.0,  10.0 };

            // Целевая функция: f(x, y) = x^2 + y^2
            Func<double[], double[]> objectiveFunction = position =>
            {
                double x = position[0];
                double y = position[1];
                double value = x * x + y * y;
                return new[] { value }; // вектор целей (здесь одна цель)
            };

            // ---------- Создаём оптимизатор ----------
            var optimizer = new MOPSO(
                swarmSize: 30,
                dimensions: dimensions,
                objectives: objectives,
                lowerBounds: lowerBounds,
                upperBounds: upperBounds,
                objectiveFunction: objectiveFunction,
                maxArchiveSize: 100
            );

            optimizer.Initialize();
            optimizer.Iterate(iterations: 200);

            // ---------- Результат ----------
            var best = optimizer.GetBestSolution();
            if (best != null)
            {
                Console.WriteLine("Лучшее найденное решение (по первой цели):");
                Console.WriteLine($"x = {best.Position[0]:F6}");
                Console.WriteLine($"y = {best.Position[1]:F6}");
                Console.WriteLine($"f(x, y) = {best.Objectives[0]:F6}");
            }
            else
            {
                Console.WriteLine("Архив пуст, решение не найдено.");
            }

            // Чтобы окно не закрывалось сразу (если запускаешь из IDE)
            // Console.ReadKey();
        }
    }

    // ---------------- Модели частиц и архива ----------------

    public class Particle
    {
        public double[] Position;
        public double[] Velocity;
        public double[] BestPosition;    // личное лучшее положение
        public double[] BestObjectives;  // значения целей в лучшем положении
        public double[] Objectives;      // текущие значения целей

        public Particle(int dimensions, int objectives)
        {
            Position = new double[dimensions];
            Velocity = new double[dimensions];
            BestPosition = new double[dimensions];
            Objectives = new double[objectives];
            BestObjectives = null; // ещё не задано
        }
    }

    public class ArchiveMember
    {
        public double[] Position;
        public double[] Objectives;
        public double CrowdingDistance;

        public ArchiveMember(double[] position, double[] objectives)
        {
            Position = position;
            Objectives = objectives;
            CrowdingDistance = 0.0;
        }

        public ArchiveMember Clone()
        {
            return new ArchiveMember(
                (double[])Position.Clone(),
                (double[])Objectives.Clone()
            );
        }
    }

    // ---------------- Реализация MOPSO ----------------

    public class MOPSO
    {
        private readonly int swarmSize;
        private readonly int dimensions;
        private readonly int objectives;
        private readonly double[] lowerBounds;
        private readonly double[] upperBounds;
        private readonly Func<double[], double[]> objectiveFunction;
        private readonly int maxArchiveSize;

        private readonly Random random = new Random();

        private readonly List<Particle> swarm = new List<Particle>();
        private List<ArchiveMember> archive = new List<ArchiveMember>();

        // Параметры PSO
        private readonly double inertiaWeight = 0.7;
        private readonly double cognitiveCoeff = 1.5;
        private readonly double socialCoeff = 1.5;

        public MOPSO(
            int swarmSize,
            int dimensions,
            int objectives,
            double[] lowerBounds,
            double[] upperBounds,
            Func<double[], double[]> objectiveFunction,
            int maxArchiveSize = 100)
        {
            if (lowerBounds.Length != dimensions || upperBounds.Length != dimensions)
                throw new ArgumentException("Размерность границ должна совпадать с числом измерений.");

            this.swarmSize = swarmSize;
            this.dimensions = dimensions;
            this.objectives = objectives;
            this.lowerBounds = lowerBounds;
            this.upperBounds = upperBounds;
            this.objectiveFunction = objectiveFunction;
            this.maxArchiveSize = maxArchiveSize;
        }

        public void Initialize()
        {
            swarm.Clear();
            archive.Clear();

            for (int i = 0; i < swarmSize; i++)
            {
                var p = new Particle(dimensions, objectives);

                for (int d = 0; d < dimensions; d++)
                {
                    double range = upperBounds[d] - lowerBounds[d];
                    p.Position[d] = lowerBounds[d] + random.NextDouble() * range;
                    // небольшая начальная скорость
                    p.Velocity[d] = (random.NextDouble() - 0.5) * range * 0.1;
                }

                p.Objectives = objectiveFunction(p.Position);
                p.BestPosition = (double[])p.Position.Clone();
                p.BestObjectives = (double[])p.Objectives.Clone();

                swarm.Add(p);
            }

            UpdateArchiveFromSwarm();
        }

        public void Iterate(int iterations)
        {
            for (int iter = 0; iter < iterations; iter++)
            {
                // 1. Пересчитываем цели и обновляем личные best с учётом доминирования
                foreach (var p in swarm)
                {
                    p.Objectives = objectiveFunction(p.Position);

                    if (p.BestObjectives == null)
                    {
                        p.BestObjectives = (double[])p.Objectives.Clone();
                        p.BestPosition = (double[])p.Position.Clone();
                    }
                    else
                    {
                        bool currentDominatesBest = Dominates(p.Objectives, p.BestObjectives);
                        bool bestDominatesCurrent = Dominates(p.BestObjectives, p.Objectives);

                        if (currentDominatesBest)
                        {
                            p.BestObjectives = (double[])p.Objectives.Clone();
                            p.BestPosition = (double[])p.Position.Clone();
                        }
                        else if (!bestDominatesCurrent)
                        {
                            // Ни одно не доминирует другое — случайно обновляем
                            if (random.NextDouble() < 0.5)
                            {
                                p.BestObjectives = (double[])p.Objectives.Clone();
                                p.BestPosition = (double[])p.Position.Clone();
                            }
                        }
                    }
                }

                // 2. Обновляем внешний архив недоминируемых решений
                UpdateArchiveFromSwarm();

                // 3. Обновляем скорости и положения
                foreach (var p in swarm)
                {
                    var leader = SelectLeader();

                    for (int d = 0; d < dimensions; d++)
                    {
                        double r1 = random.NextDouble();
                        double r2 = random.NextDouble();

                        double cognitive = cognitiveCoeff * r1 * (p.BestPosition[d] - p.Position[d]);
                        double social = socialCoeff * r2 * (leader.Position[d] - p.Position[d]);

                        p.Velocity[d] = inertiaWeight * p.Velocity[d] + cognitive + social;

                        p.Position[d] += p.Velocity[d];

                        // Ограничиваем положение в заданных пределах
                        if (p.Position[d] < lowerBounds[d])
                        {
                            p.Position[d] = lowerBounds[d];
                            p.Velocity[d] *= -0.5;
                        }
                        else if (p.Position[d] > upperBounds[d])
                        {
                            p.Position[d] = upperBounds[d];
                            p.Velocity[d] *= -0.5;
                        }
                    }
                }
            }
        }

        // --------- Работа с архивом и лидером ---------

        private void UpdateArchiveFromSwarm()
        {
            var candidates = new List<ArchiveMember>();

            // старый архив
            foreach (var a in archive)
                candidates.Add(a.Clone());

            // личные best частиц
            foreach (var p in swarm)
            {
                if (p.BestObjectives == null) continue;

                candidates.Add(new ArchiveMember(
                    (double[])p.BestPosition.Clone(),
                    (double[])p.BestObjectives.Clone()
                ));
            }

            // оставляем только недоминируемые решения
            var newArchive = new List<ArchiveMember>();
            for (int i = 0; i < candidates.Count; i++)
            {
                bool dominated = false;
                for (int j = 0; j < candidates.Count; j++)
                {
                    if (i == j) continue;
                    if (Dominates(candidates[j].Objectives, candidates[i].Objectives))
                    {
                        dominated = true;
                        break;
                    }
                }
                if (!dominated)
                    newArchive.Add(candidates[i]);
            }

            // если архив слишком большой — обрезаем по crowding distance
            if (newArchive.Count > maxArchiveSize)
            {
                ComputeCrowdingDistance(newArchive);
                newArchive = newArchive
                    .OrderByDescending(a => a.CrowdingDistance)
                    .Take(maxArchiveSize)
                    .ToList();
            }

            archive = newArchive;
        }

        private ArchiveMember SelectLeader()
        {
            if (archive.Count == 0)
                throw new InvalidOperationException("Архив пуст, невозможно выбрать лидера.");

            ComputeCrowdingDistance(archive);

            var sorted = archive
                .OrderByDescending(a => a.CrowdingDistance)
                .ToList();

            int maxIndex = Math.Max(1, sorted.Count / 2);
            int index = random.Next(maxIndex);
            return sorted[index];
        }

        // Берём лучшее по первой цели (в нашем тесте цель одна)
        public ArchiveMember GetBestSolution()
        {
            if (archive.Count == 0)
                return null;

            return archive.OrderBy(a => a.Objectives[0]).First();
        }

        public IReadOnlyList<ArchiveMember> Archive => archive;

        // --------- Вспомогательные методы: доминирование и crowding ---------

        // A доминирует B (минимизация): лучше или равно по всем и строго лучше хотя бы по одной
        private static bool Dominates(double[] a, double[] b)
        {
            bool strictlyBetter = false;

            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] > b[i]) return false;
                if (a[i] < b[i]) strictlyBetter = true;
            }

            return strictlyBetter;
        }

        private static void ComputeCrowdingDistance(List<ArchiveMember> archive)
        {
            int n = archive.Count;
            if (n == 0) return;

            int m = archive[0].Objectives.Length;

            foreach (var a in archive)
                a.CrowdingDistance = 0.0;

            if (n <= 2)
            {
                foreach (var a in archive)
                    a.CrowdingDistance = double.PositiveInfinity;
                return;
            }

            for (int obj = 0; obj < m; obj++)
            {
                archive.Sort((a, b) => a.Objectives[obj].CompareTo(b.Objectives[obj]));

                double min = archive[0].Objectives[obj];
                double max = archive[n - 1].Objectives[obj];
                double range = max - min;
                if (range == 0.0) range = 1.0;

                archive[0].CrowdingDistance = double.PositiveInfinity;
                archive[n - 1].CrowdingDistance = double.PositiveInfinity;

                for (int i = 1; i < n - 1; i++)
                {
                    double prev = archive[i - 1].Objectives[obj];
                    double next = archive[i + 1].Objectives[obj];

                    archive[i].CrowdingDistance += (next - prev) / range;
                }
            }
        }
    }
}

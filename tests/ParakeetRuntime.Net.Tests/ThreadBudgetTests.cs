using ParakeetRuntime.Net;

namespace ParakeetRuntime.Net.Tests;

public sealed class ThreadBudgetTests
{
    private static IEnumerable<CpuCore> Cores(int count, int efficiencyClass, int threadsPerCore = 1, int firstCore = 0) =>
        Enumerable.Range(firstCore, count).SelectMany(core => Enumerable.Repeat(new CpuCore(0, core, efficiencyClass), threadsPerCore));

    [Fact]
    public void A_hybrid_CPU_gets_one_thread_per_performance_core()
    {
        var coreUltra7_270KPlus = Cores(8, efficiencyClass: 1).Concat(Cores(16, efficiencyClass: 0, firstCore: 8));

        Assert.Equal(8, ThreadBudget.For(coreUltra7_270KPlus));
    }

    [Fact]
    public void Hyper_threads_of_one_core_count_once()
    {
        var i7_11850H = Cores(8, efficiencyClass: 0, threadsPerCore: 2);

        Assert.Equal(8, ThreadBudget.For(i7_11850H));
    }

    [Fact]
    public void A_small_CPU_gets_a_thread_per_core()
    {
        Assert.Equal(4, ThreadBudget.For(Cores(4, efficiencyClass: 0, threadsPerCore: 2)));
    }

    [Fact]
    public void A_large_CPU_leaves_cores_beyond_eight_to_the_app()
    {
        Assert.Equal(ThreadBudget.Most, ThreadBudget.For(Cores(16, efficiencyClass: 0, threadsPerCore: 2)));
    }

    [Fact]
    public void Pinned_to_efficiency_cores_it_uses_those_cores()
    {
        var allowedEfficiencyCoresOnly = Cores(4, efficiencyClass: 0);

        Assert.Equal(4, ThreadBudget.For(allowedEfficiencyCoresOnly));
    }

    [Fact]
    public void Without_core_information_it_keeps_four_threads()
    {
        Assert.Equal(Math.Min(4, Environment.ProcessorCount), ThreadBudget.For([]));
    }
}

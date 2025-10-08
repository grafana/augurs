package outlier

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDetect(t *testing.T) {
	t.Parallel()

	t.Run("dbscan", func(t *testing.T) {
		ctx := t.Context()
		f, err := NewOutlierFactory(ctx, nil)
		require.NoError(t, err)
		t.Cleanup(func() { f.Close(ctx) })

		ins, err := f.Instantiate(ctx)
		require.NoError(t, err)
		t.Cleanup(func() { ins.Close(ctx) })

		input := Input{
			Algorithm: AlgorithmDbscan{
				EpsilonOrSensitivity: EpsilonOrSensitivitySensitivity(0.5),
			},
			Data: [][]float64{
				{1.0, 2.0, 1.5, 2.3},
				{1.9, 2.2, 1.2, 2.4},
				{1.5, 2.1, 6.4, 8.5},
			},
		}

		detected, err := ins.Detect(ctx, input)
		require.NoError(t, err)

		assert.Len(t, detected.OutlyingSeries, 1)
		assert.Contains(t, detected.OutlyingSeries, uint32(2))
		assert.True(t, detected.SeriesResults[2].IsOutlier)
		assert.Equal(t, []float64{0.0, 0.0, 1.0, 1.0}, detected.SeriesResults[2].Scores)
		assert.NotNil(t, detected.ClusterBand)
	})

	t.Run("mad", func(t *testing.T) {
		t.Parallel()
		ctx := t.Context()
		f, err := NewOutlierFactory(ctx, nil)
		require.NoError(t, err)
		t.Cleanup(func() { f.Close(ctx) })

		ins, err := f.Instantiate(ctx)
		require.NoError(t, err)
		t.Cleanup(func() { ins.Close(ctx) })

		input := Input{
			Algorithm: AlgorithmMad{
				ThresholdOrSensitivity: ThresholdOrSensitivitySensitivity(0.5),
			},
			Data: [][]float64{
				{31.6},
				{33.12},
				{33.84},
				{100.234},
				{12.83},
				{15.23},
				{33.23},
				{32.85},
				{24.72},
			},
		}
		detected, err := ins.Detect(ctx, input)
		require.NoError(t, err)

		assert.Len(t, detected.OutlyingSeries, 1)
		assert.Contains(t, detected.OutlyingSeries, uint32(3))
		assert.True(t, detected.SeriesResults[3].IsOutlier)
		assert.NotNil(t, detected.ClusterBand)
	})
}

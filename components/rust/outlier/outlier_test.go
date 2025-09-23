package outlier

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type algorithmEpsilonOrSensitivity any

type algorithmEpsilonOrSensitivitySensitivity struct {
	Sensitivity float64 `json:"sensitivity"`
}

type algorithmEpsilonOrSensitivityEpsilon struct {
	Epsilon float64 `json:"epsilon"`
}

type algorithm any

type algorithmDBSCAN struct {
	EpsilonOrSensitivity algorithmEpsilonOrSensitivity `json:"epsilonOrSensitivity"`
}

type algorithmThresholdOrSensitivity any
type algorithmThresholdOrSensitivitySensitivity struct {
	Sensitivity float64 `json:"sensitivity"`
}
type algorithmThresholdOrSensitivityThreshold struct {
	Threshold float64 `json:"threshold"`
}

type algorithmMAD struct {
	ThresholdOrSensitivity algorithmThresholdOrSensitivity `json:"thresholdOrSensitivity"`
}

type input struct {
	Algorithm algorithm   `json:"algorithm"`
	Data      [][]float64 `json:"data"`
}

type outlierInterval struct {
	Start int  `json:"start"`
	End   *int `json:"end"`
}

type series struct {
	IsOutlier        bool              `json:"isOutlier"`
	OutlierIntervals []outlierInterval `json:"outlierIntervals"`
	Scores           []float64         `json:"scores"`
}

type band struct {
	Min []float64 `json:"min"`
	Max []float64 `json:"max"`
}

type output struct {
	OutlyingSeries []int    `json:"outlyingSeries"`
	SeriesResults  []series `json:"seriesResults"`
	ClusterBand    *band    `json:"clusterBand"`
}

func TestDetect(t *testing.T) {
	t.Parallel()

	t.Run("dbscan", func(t *testing.T) {
		ctx := t.Context()
		f, err := NewOutlierFactory(ctx)
		require.NoError(t, err)
		t.Cleanup(func() { f.Close(ctx) })

		ins, err := f.Instantiate(ctx)
		require.NoError(t, err)
		t.Cleanup(func() { ins.Close(ctx) })

		input := input{
			Algorithm: algorithmDBSCAN{
				EpsilonOrSensitivity: algorithmEpsilonOrSensitivitySensitivity{
					Sensitivity: 0.5,
				},
			},
			Data: [][]float64{
				{1.0, 2.0, 1.5, 2.3},
				{1.9, 2.2, 1.2, 2.4},
				{1.5, 2.1, 6.4, 8.5},
			},
		}
		s, err := json.Marshal(input)
		require.NoError(t, err)
		inputStr := string(s)

		detectedStr, err := ins.Detect(ctx, inputStr)
		require.NoError(t, err)

		var detected output
		err = json.Unmarshal([]byte(detectedStr), &detected)
		require.NoError(t, err)

		assert.Len(t, detected.OutlyingSeries, 1)
		assert.Contains(t, detected.OutlyingSeries, 2)
		assert.True(t, detected.SeriesResults[2].IsOutlier)
		assert.Equal(t, []float64{0.0, 0.0, 1.0, 1.0}, detected.SeriesResults[2].Scores)
		assert.NotNil(t, detected.ClusterBand)
	})

	t.Run("mad", func(t *testing.T) {
		t.Parallel()
		ctx := t.Context()
		f, err := NewOutlierFactory(ctx)
		require.NoError(t, err)
		t.Cleanup(func() { f.Close(ctx) })

		ins, err := f.Instantiate(ctx)
		require.NoError(t, err)
		t.Cleanup(func() { ins.Close(ctx) })

		input := input{
			Algorithm: algorithmMAD{
				ThresholdOrSensitivity: algorithmThresholdOrSensitivitySensitivity{
					Sensitivity: 0.5,
				},
			},
			Data: [][]float64{
				{31.6},
				{33.12},
				{33.84},
				{38.234}, // outlier
				{12.83},
				{15.23},
				{33.23},
				{32.85},
				{24.72},
			},
		}
		s, err := json.Marshal(input)
		require.NoError(t, err)
		inputStr := string(s)

		detectedStr, err := ins.Detect(ctx, inputStr)
		require.NoError(t, err)

		var detected output
		err = json.Unmarshal([]byte(detectedStr), &detected)
		require.NoError(t, err)

		assert.Len(t, detected.OutlyingSeries, 1)
		assert.Contains(t, detected.OutlyingSeries, 3)
		assert.True(t, detected.SeriesResults[3].IsOutlier)
		assert.NotNil(t, detected.ClusterBand)
	})
}

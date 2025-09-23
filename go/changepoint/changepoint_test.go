package changepoint

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDetect(t *testing.T) {
	t.Parallel()

	t.Run("argpcp", func(t *testing.T) {
		ctx := t.Context()
		f, err := NewChangepointFactory(ctx, nil)
		require.NoError(t, err)
		t.Cleanup(func() { f.Close(ctx) })

		ins, err := f.Instantiate(ctx)
		require.NoError(t, err)
		t.Cleanup(func() { ins.Close(ctx) })

		input := Input{
			Data:      []float64{0.5, 1.0, 0.4, 0.8, 1.5, 0.9, 0.6, 25.3, 20.4, 27.3, 30.0},
			Algorithm: AlgorithmArgpcp{},
		}
		cps, err := ins.Detect(ctx, input)
		require.NoError(t, err)
		assert.Equal(t, []uint32{0, 6}, cps)
	})

	t.Run("bocpd", func(t *testing.T) {
		t.Parallel()
		ctx := t.Context()
		f, err := NewChangepointFactory(ctx, nil)
		require.NoError(t, err)
		t.Cleanup(func() { f.Close(ctx) })

		ins, err := f.Instantiate(ctx)
		require.NoError(t, err)
		t.Cleanup(func() { ins.Close(ctx) })

		input := Input{
			Data:      []float64{0.5, 1.0, 0.4, 0.8, 1.5, 0.9, 0.6, 25.3, 20.4, 27.3, 30.0},
			Algorithm: AlgorithmBocpd{},
		}
		cps, err := ins.Detect(ctx, input)
		require.NoError(t, err)
		assert.Equal(t, []uint32{0, 6}, cps)
	})
}

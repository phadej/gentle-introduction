{-# LANGUAGE ConstraintKinds #-}
-- | /Gentle introduction/ is a collection of non-clashing names
-- from Hackage.
--
-- We try to be explicit in exports.
-- In particular with "Prelude" and for type-classes which members have changed
-- in different @base@ versions. This is one reason to keep base support window
-- relatively short.
--
module Gentle.Introduction (
    -- * We start with a "Prelude"
    -- ** Equality
    Eq (..), Ord (..), Ordering (..),
    -- ** "lambda calculus"
    -- | Basic calculus types and combinators on them
    Either (..), Maybe (..), const, curry, either, flip, fst, id, maybe, snd, uncurry, ($), (.),
    -- ** Functor, Monad
    -- | 'Applicative' is imported from "Control.Applicative". 'Monad' doesn't have @fail@ ('MonadFail' does).
    Functor (fmap, (<$)), Monad (return, (>>=)), (<$>),
    -- ** List
    -- | ... almost
    concat, concatMap, drop, dropWhile, lines, map, replicate, reverse, span, take, takeWhile, unlines, unwords, words, (++),
    -- ** Booleans
    Bool (..), not, otherwise, (&&), (||),
    -- ** Numerics
    Double, Fractional (..), Int, Integer, Integral (..), Num (..), Rational, Real (..), RealFrac (..), Word, fromIntegral, realToFrac,
    -- ** Operational
    error, seq, undefined, ($!),
    -- ** Others
    -- | Some mixed stuff
    Bounded (minBound, maxBound), Char, Enum (succ,pred,toEnum,fromEnum), FilePath, IO, Show (..), String, print,
    -- * Algebra.Lattice
    Lattice ((/\), (\/)),
    -- * Control.Applicative
    Applicative (pure, (<*>), (<*), (*>), liftA2), Alternative (empty, (<|>), some, many), optional,
    -- * Control.Concurrent.STM
    STM, atomically,
    -- * Control.DeepSeq
    NFData (rnf),
    force, deepseq,
    -- * Control.Exception
    SomeException (..), Exception (toException, fromException, displayException),
    -- * Control.Monad
    ap,
    join,
    when, unless,
    void,
    guard,
    -- * Control.Monad.Error.Class
    MonadError (throwError, catchError),
    -- * Control.Monad.Fail
    MonadFail (fail),
    -- * Control.Monad.IO.Class
    MonadIO (liftIO),
    -- * Control.Monad.IO.Unlift
    MonadUnliftIO (..),
    -- * Data.Bifoldable
    Bifoldable (..),
    -- * Data.Bifunctor
    Bifunctor (..),
    -- * Data.Bifunctor.Assoc
    Assoc (..),
    -- * Data.Bifunctor.Swap
    Swap (..),
    -- * Data.Bitraversable
    Bitraversable (..),
    bimapDefault, bifoldMapDefault,
    -- * Data.Bytestring
    ByteString,
    -- * Data.CaseInsensitive
    CI,
    -- * Data.Char
    isSpace, isLower, isUpper, isAlpha, isAlphaNum, isNumber,
    isAsciiLower, isAsciiUpper, isAsciiAlpha, isAsciiAlphaNum, isDigit,
    -- * Data.Coerce
    Coercible, coerce,
    -- * Data.Fix
    Fix (..),
    -- * Data.Foldable
    -- | Unfortunately GHC-8.6.5 base doesn't have 'foldMap''.
    --
    Foldable (foldMap, foldr, foldr', foldl, foldl', toList, null, length, elem),
    notElem,
    for_, traverse_,
    and, or,
    all, any,
    -- * Data.Foldable.WithIndex
    FoldableWithIndex (..), ifor_, itraverse_, itoList,
    -- * Data.Function
    (&),
    -- * Data.Functor
    (<&>),
    -- * Data.Functor.Compose
    Compose (..),
    -- * Data.Functor.Const
    Const (..),
    -- * Data.Functor.Contravariant
    Contravariant (..), (>$<),
    -- * Data.Functor.Identity
    Identity (..),
    -- * Data.Functor.WithIndex
    FunctorWithIndex (..),
    -- * Data.Hashable
    Hashable,
    -- * Data.Int
    Int8, Int16, Int32, Int64,
    -- * Data.Kind
    Type, Constraint,
    -- * Data.List
    chunksOf, dropWhileEnd,
    -- * Data.List.NonEmpty
    NonEmpty (..),
    groupBy, head, last, some1,
    -- * Data.Map
    Map,
    -- * Data.Maybe
    fromMaybe, isJust, isNothing,
    maybeToList, listToMaybe,
    -- * Data.Monoid
    Monoid (mappend, mempty, mconcat),
    -- * Data.Proxy
    Proxy (..),
    -- * Data.Scientific
    Scientific,
    -- * Data.Semialign
    Semialign (alignWith, align), Zip (zipWith, zip), Align (nil), Repeat (repeat),
    -- * Data.Semigroup
    Semigroup ((<>)),
    gmempty, gmappend,
    -- * Data.Set,
    Set,
    -- * Data.Show
    tshow,
    -- * Data.Some
    Some (..),
    -- * Data.SOP
    I (..), K (..), NP (..), NS (..),
    SListI,
    -- * Data.Strict
    Strict (toStrict, toLazy),
    -- * Data.String
    IsString (..),
    -- * Data.Tagged
    Tagged (..),
    -- * Data.Text
    Text,
    -- * Data.Text.Encoding
    TE.encodeUtf8, decodeUtf8Lenient,
    -- * Data.These
    These (..),
    -- * Data.Time.Compat
    UTCTime, Day,
    -- * Data.Traversable
    Traversable (traverse, sequenceA),
    for,
    fmapDefault, foldMapDefault,
    -- * Data.Traversable.WithIndex
    TraversableWithIndex (..), ifor,
    -- * Data.Type.Equality
    (:~:) (..),
    -- * Data.Universe.Class
    Universe (..), Finite (..),
    -- * Data.UUID.Types
    UUID,
    -- * Data.Void
    Void, absurd,
    -- * Data.Word
    Word8, Word16, Word32, Word64,
    -- * Witherable
    Filterable (..), FilterableWithIndex (..),
    ordNub, ordNubOn, hashNub, hashNubOn,
    -- * Debug.Trace
    -- | These functions are marked as deprecated
    traceShow, traceShowId,
    -- * GHC.Stack
    HasCallStack,
    -- * Numeric.Natural
    Natural,
    -- * Text.Read
    readMaybe,
    -- * Type.Reflection
    Typeable, TypeRep, typeRep,
    -- * GHC.Generic
    Generic, Generic1,
    -- * UTF8 conversion functions
    fromUTF8BS, fromUTF8LBS,
    toUTF8BS, toUTF8LBS,
) where

-------------------------------------------------------------------------------
-- Prelude
-------------------------------------------------------------------------------

-- Eq and Ord
import Prelude (Eq (..), Ord (..), Ordering (..))

-- "lambda calculus"
import Prelude (Either (..), Maybe (..), const, curry, either, flip, fst, id, maybe, snd, uncurry, ($), (.))

-- functor, applicative, monad
import Prelude (Functor (..), Monad (return, (>>=)), (<$>))

-- list
import Prelude (concat, concatMap, drop, dropWhile, lines, map, replicate, reverse, span, take, takeWhile, unlines, unwords, words, (++))

-- boolean
import Prelude (Bool (..), not, otherwise, (&&), (||))

-- numeric
import Prelude
       (Double, Fractional (..), Int, Integer, Integral (..), Num (..), Rational, Real (..), RealFrac (..), Word, fromIntegral, realToFrac)

-- operational
import Prelude (error, seq, undefined, ($!))

-- rest
import Prelude (Bounded (..), Char, Enum (..), FilePath, IO, Show (..), String, print)

-------------------------------------------------------------------------------
-- Other imports
-------------------------------------------------------------------------------

import Algebra.Lattice            (Lattice (..))
import Control.Applicative        (Alternative (..), Applicative (liftA2, pure, (*>), (<*), (<*>)), liftA2, optional)
import Control.Concurrent.STM     (STM, atomically)
import Control.DeepSeq            (NFData (..), deepseq, force)
import Control.Exception          (Exception (..), SomeException (..))
import Control.Monad              (ap, guard, join, unless, void, when)
import Control.Monad.Error.Class  (MonadError (..))
import Control.Monad.Fail         (MonadFail (..))
import Control.Monad.IO.Class     (MonadIO (..))
import Control.Monad.IO.Unlift    (MonadUnliftIO (withRunInIO))
import Data.Bifoldable            (Bifoldable (..))
import Data.Bifunctor             (Bifunctor (..))
import Data.Bifunctor.Assoc       (Assoc (..))
import Data.Bifunctor.Swap        (Swap (..))
import Data.Bitraversable         (Bitraversable (..), bifoldMapDefault, bimapDefault)
import Data.ByteString            (ByteString)
import Data.CaseInsensitive       (CI)
import Data.Char                  (isAlpha, isAlphaNum, isDigit, isLower, isNumber, isSpace, isUpper)
import Data.Coerce                (Coercible, coerce)
import Data.Fix                   (Fix (..))
import Data.Foldable              (Foldable (..), all, and, any, for_, notElem, or, traverse_)
import Data.Foldable.WithIndex    (FoldableWithIndex (..), ifor_, itoList, itraverse_)
import Data.Function              ((&))
import Data.Functor               ((<&>))
import Data.Functor.Compose       (Compose (..))
import Data.Functor.Const         (Const (..))
import Data.Functor.Contravariant (Contravariant (..), (>$<))
import Data.Functor.Identity      (Identity (..))
import Data.Functor.WithIndex     (FunctorWithIndex (..))
import Data.Hashable              (Hashable)
import Data.Int                   (Int16, Int32, Int64, Int8)
import Data.Kind                  (Constraint, Type)
import Data.List                  (dropWhileEnd)
import Data.List.NonEmpty         (NonEmpty (..), groupBy, head, last, some1)
import Data.Map                   (Map)
import Data.Maybe                 (fromMaybe, isJust, isNothing, listToMaybe, maybeToList)
import Data.Monoid                (Monoid (..))
import Data.Proxy                 (Proxy (..))
import Data.Scientific            (Scientific)
import Data.Semialign             (Align (..), Repeat (..), Semialign (..), Zip (..))
import Data.Semigroup             (Semigroup (..))
import Data.Semigroup.Generic     (gmappend, gmempty)
import Data.Set                   (Set)
import Data.Some.GADT             (Some (..))
import Data.SOP                   (I (..), K (..), NP (..), NS (..), SListI)
import Data.Strict                (Strict (..))
import Data.String                (IsString (..))
import Data.Tagged                (Tagged (..))
import Data.Text                  (Text)
import Data.These                 (These (..))
import Data.Time.Compat           (Day, UTCTime)
import Data.Traversable           (Traversable (..), fmapDefault, foldMapDefault, for)
import Data.Traversable.WithIndex (TraversableWithIndex (..), ifor)
import Data.Type.Equality         ((:~:) (..))
import Data.Universe.Class        (Finite (..), Universe (..))
import Data.UUID.Types            (UUID)
import Data.Void                  (Void, absurd)
import Data.Word                  (Word16, Word32, Word64, Word8)
import GHC.Generics               (Generic, Generic1)
import GHC.Stack                  (HasCallStack)
import Numeric.Natural            (Natural)
import Text.Read                  (readMaybe)
import Type.Reflection            (TypeRep, Typeable, typeRep)
import Witherable                 (Filterable (..), FilterableWithIndex (..), hashNub, hashNubOn, ordNub, ordNubOn)

import Distribution.Utils.Generic (fromUTF8BS, fromUTF8LBS, toUTF8BS, toUTF8LBS)

import qualified Data.Text                as T
import qualified Data.Text.Encoding       as TE
import qualified Data.Text.Encoding.Error as TEE
import qualified Debug.Trace              as Trace

import Data.Functor.WithIndex.Instances ()
import Data.Orphans ()

import qualified Prelude as P

-------------------------------------------------------------------------------
-- Data.Char
-------------------------------------------------------------------------------

-- | @[a-z]@
isAsciiLower :: Char -> Bool
isAsciiLower c = 'a' <= c && c <= 'z'

-- | @[A-Z]@
isAsciiUpper :: Char -> Bool
isAsciiUpper c = 'A' <= c && c <= 'Z'

-- | @[a-zA-Z]@
isAsciiAlpha :: Char -> Bool
isAsciiAlpha c = isAsciiLower c || isAsciiUpper c

-- | @[a-zA-Z0-9]@
isAsciiAlphaNum :: Char -> Bool
isAsciiAlphaNum c = isAsciiAlpha c || isDigit c

-------------------------------------------------------------------------------
-- Data.List
-------------------------------------------------------------------------------

-- |
--
-- >>> chunksOf 3 [1..10]
-- [[1,2,3],[4,5,6],[7,8,9],[10]]
--
chunksOf :: Int -> [a] -> [[a]]
chunksOf n = go where
    go [] = []
    go xs = case P.splitAt n xs of
        ~(ys,zs) -> ys : go zs

-------------------------------------------------------------------------------
-- Data.Show
-------------------------------------------------------------------------------

-- | @'T.pack' . 'show'@
--
-- >>> tshow (Just 'x')
-- "Just 'x'"
--
tshow :: Show a => a -> Text
tshow = T.pack . show

-------------------------------------------------------------------------------
-- Data.Text.Encoding
-------------------------------------------------------------------------------

-- | A @decodeUtf8@ variant, which doesn't 'error'.
decodeUtf8Lenient :: ByteString -> Text
decodeUtf8Lenient = TE.decodeUtf8With TEE.lenientDecode

-------------------------------------------------------------------------------
-- Debug.Trace versions
-------------------------------------------------------------------------------

traceShow :: Show a => a -> b -> b
traceShow = Trace.traceShow
{-# DEPRECATED traceShow "Don't leave me here" #-}

traceShowId :: Show a => a -> a
traceShowId = Trace.traceShowId
{-# DEPRECATED traceShowId "Don't leave me here" #-}
